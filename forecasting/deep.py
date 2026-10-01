"""Neural forecasters: LSTM, temporal CNN, and a patch Transformer.

These are the expensive arm of the model pool. On a CPU-only machine they cost
seconds rather than milliseconds per fit, so the experiment config restricts them to a
subsample of instances (``forecast.heavy_models`` / ``heavy_model_fraction``) and the
manuscript reports that restriction rather than hiding it.

Three conventions apply to all three architectures, so that any difference in results
is attributable to the architecture and not to the training setup:

* **Direct multi-step output.** The network emits all ``horizon`` values at once.
  Unlike the recursive tree baselines this avoids compounding error, and unlike a
  per-step model it costs one fit.
* **Instance normalisation.** Each input window is z-scored using its own statistics,
  and the prediction is de-normalised afterwards. This is what makes a single small
  network work across series whose levels differ by orders of magnitude, and it is
  standard in recent forecasting architectures.
* **Identical training loop.** Same optimiser, epoch budget, early stopping and loss.

Deliberately small: with a few hundred training windows, a larger network would overfit
and would misrepresent what deep models offer at this data scale. The paper's claim is
not that these are state-of-the-art implementations -- it is that they are trained
identically and fairly.
"""

from __future__ import annotations

import numpy as np

from forecasting.base import FORECASTERS, Forecaster, residual_samples

__all__ = ["LSTMForecaster", "TCNForecaster", "TransformerForecaster"]

_MAX_EPOCHS = 80
_PATIENCE = 12
_BATCH = 32


def _windows(x: np.ndarray, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding (input, target) windows. Raises if the history is too short."""
    n = x.size
    n_win = n - lookback - horizon + 1
    if n_win < 4:
        raise ValueError(
            f"need at least 4 windows, got {n_win} "
            f"(n={n}, lookback={lookback}, horizon={horizon})"
        )
    X = np.empty((n_win, lookback), dtype=np.float32)
    Y = np.empty((n_win, horizon), dtype=np.float32)
    for i in range(n_win):
        X[i] = x[i : i + lookback]
        Y[i] = x[i + lookback : i + lookback + horizon]
    return X, Y


def _instance_norm(X: np.ndarray, Y: np.ndarray | None = None):
    """Z-score each window by its own mean/std; return stats for de-normalisation."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xn = (X - mu) / sd
    Yn = None if Y is None else (Y - mu) / sd
    return Xn, Yn, mu, sd


class _TorchForecaster(Forecaster):
    """Shared training loop for the neural models."""

    name = "deep_base"
    cost = 20.0

    def _build(self, lookback: int, horizon: int):  # pragma: no cover - abstract
        raise NotImplementedError

    def _lookback(self, n: int, seasonal_period: int, horizon: int) -> int:
        target = max(2 * seasonal_period, 3 * horizon, 16)
        # Leave room for at least a handful of training windows.
        return int(max(8, min(target, (n - horizon) // 3)))

    def _forecast(self, history, horizon, seasonal_period, rng):
        import torch
        from torch import nn

        torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
        torch.set_num_threads(1)  # parallelism is at the experiment level

        x = history.astype(np.float32)
        lookback = self._lookback(x.size, max(1, int(seasonal_period)), horizon)
        X, Y = _windows(x, lookback, horizon)
        Xn, Yn, _, _ = _instance_norm(X, Y)

        # Chronological holdout for early stopping -- never shuffled, so the
        # validation windows are strictly later than the training windows.
        n_val = max(1, int(0.2 * len(Xn)))
        Xtr, Ytr = Xn[:-n_val], Yn[:-n_val]
        Xva, Yva = Xn[-n_val:], Yn[-n_val:]
        if len(Xtr) < 2:
            Xtr, Ytr = Xn, Yn
            Xva, Yva = Xn, Yn

        Xtr_t = torch.from_numpy(Xtr).unsqueeze(-1)
        Ytr_t = torch.from_numpy(Ytr)
        Xva_t = torch.from_numpy(Xva).unsqueeze(-1)
        Yva_t = torch.from_numpy(Yva)

        model = self._build(lookback, horizon)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        best_loss, bad = float("inf"), 0
        n_train = len(Xtr_t)

        for _ in range(_MAX_EPOCHS):
            model.train()
            perm = torch.randperm(n_train)
            for start in range(0, n_train, _BATCH):
                idx = perm[start : start + _BATCH]
                opt.zero_grad()
                loss = loss_fn(model(Xtr_t[idx]), Ytr_t[idx])
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(Xva_t), Yva_t))
            if val_loss < best_loss - 1e-6:
                best_loss, bad = val_loss, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= _PATIENCE:
                    break

        model.load_state_dict(best_state)
        model.eval()

        # --- forecast from the final window ---
        last = x[-lookback:][None, :]
        last_n, _, mu, sd = _instance_norm(last)
        with torch.no_grad():
            pred_n = model(torch.from_numpy(last_n).unsqueeze(-1)).numpy()
        point = (pred_n * sd + mu).reshape(-1).astype(float)

        # In-sample residuals on the held-out windows, de-normalised, for intervals.
        with torch.no_grad():
            fit_n = model(Xva_t).numpy()
        _, _, mu_v, sd_v = _instance_norm(X[-len(Xva) :])
        resid = ((fit_n - Yva) * sd_v).reshape(-1)
        samples = residual_samples(point, resid, rng, accumulate=False)
        return point, samples


@FORECASTERS.register("lstm")
class LSTMForecaster(_TorchForecaster):
    """Single-layer LSTM encoder with a linear multi-step head."""

    name = "lstm"
    cost = 25.0

    def _build(self, lookback, horizon):
        import torch
        from torch import nn

        hidden = int(self.params.get("hidden", 32))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.LSTM(1, hidden, num_layers=1, batch_first=True)
                self.head = nn.Linear(hidden, horizon)

            def forward(self, z):
                out, _ = self.rnn(z)
                return self.head(out[:, -1, :])

        return Net()


@FORECASTERS.register("tcn")
class TCNForecaster(_TorchForecaster):
    """Dilated causal convolution stack (temporal convolutional network)."""

    name = "tcn"
    cost = 15.0

    def _build(self, lookback, horizon):
        from torch import nn

        ch = int(self.params.get("channels", 24))
        levels = int(self.params.get("levels", 3))

        layers: list[nn.Module] = []
        in_ch = 1
        for i in range(levels):
            dilation = 2**i
            # Left-pad only: no information may flow backwards in time.
            layers += [
                nn.ConstantPad1d((2 * dilation, 0), 0.0),
                nn.Conv1d(in_ch, ch, kernel_size=3, dilation=dilation),
                nn.ReLU(),
            ]
            in_ch = ch

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.Sequential(*layers)
                self.head = nn.Linear(ch, horizon)

            def forward(self, z):
                h = self.body(z.transpose(1, 2))
                return self.head(h[:, :, -1])

        return Net()


@FORECASTERS.register("transformer")
class TransformerForecaster(_TorchForecaster):
    """Patch-based Transformer encoder in the spirit of PatchTST.

    Patching keeps the attention sequence short, which matters a great deal on CPU:
    attention over raw timesteps would dominate the runtime for no accuracy gain at
    this data scale.
    """

    name = "transformer"
    cost = 30.0

    def _build(self, lookback, horizon):
        import torch
        from torch import nn

        d_model = int(self.params.get("d_model", 32))
        n_heads = int(self.params.get("n_heads", 2))
        patch = int(self.params.get("patch", 4))
        n_patches = max(1, lookback // patch)
        used = n_patches * patch

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch, self.used, self.n_patches = patch, used, n_patches
                self.embed = nn.Linear(patch, d_model)
                self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=2 * d_model,
                    batch_first=True,
                    dropout=0.0,
                )
                self.enc = nn.TransformerEncoder(layer, num_layers=2)
                self.head = nn.Linear(d_model * n_patches, horizon)

            def forward(self, z):
                b = z.shape[0]
                # Keep the most recent `used` steps so patches align to the present.
                seq = z[:, -self.used :, 0].reshape(b, self.n_patches, self.patch)
                h = self.enc(self.embed(seq) + self.pos)
                return self.head(h.reshape(b, -1))

        return Net()
