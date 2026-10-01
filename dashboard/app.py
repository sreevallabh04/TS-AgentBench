"""TS-AgentBench research dashboard.

    streamlit run dashboard/app.py

Deliberately plain. The dashboard exists to inspect *what the agent decided and why* --
forecasts, intervals, detected regimes and anomalies, the reliability signals, the
diagnosed failure mode and the resulting correction. It is an instrument for reading
the experiment, not a product surface, and no effort is spent on visual polish that
would not survive into the paper.

It reads completed run directories under ``results/runs/``; it never launches
experiments, so nothing here can accidentally overwrite a result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.paths import RUNS_DIR  # noqa: E402

st.set_page_config(page_title="TS-AgentBench", layout="wide")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def list_runs() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    runs = [
        p.name for p in RUNS_DIR.iterdir()
        if p.is_dir() and (p / "manifest.json").exists()
    ]
    return sorted(runs, reverse=True)


@st.cache_data(show_spinner=False)
def load_run(run_id: str) -> dict:
    import json

    directory = RUNS_DIR / run_id
    data: dict = {"manifest": json.loads((directory / "manifest.json").read_text())}
    for name in ("outcomes", "signals", "test_labels", "scores", "decisions",
                 "comparison", "realizable_oracle", "feature_importance"):
        path = directory / f"{name}.parquet"
        if path.exists():
            data[name] = pd.read_parquet(path)
    return data


@st.cache_data(show_spinner=False)
def regenerate_series(family: str, index: int, n_timesteps: int, seed: int = 0):
    """Rebuild a synthetic series from its identifier.

    Series are not stored in run directories -- they are fully determined by
    (family, index, seed), so regenerating is cheaper and cannot drift.
    """
    from datasets.synthetic import generate_family

    series = generate_family(family, index + 1, base_seed=seed, n_timesteps=n_timesteps)
    return series[index]


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

runs = list_runs()
if not runs:
    st.title("TS-AgentBench")
    st.warning(
        "No completed runs found under `results/runs/`.\n\n"
        "Produce one with:\n\n"
        "```\npython run_experiment.py --config configs/smoke.yaml\n```"
    )
    st.stop()

st.sidebar.title("TS-AgentBench")
run_id = st.sidebar.selectbox("Run", runs)
run = load_run(run_id)
manifest = run["manifest"]

st.sidebar.caption(
    f"**status:** {manifest['status']}  \n"
    f"**config:** `{manifest['config_hash'][:12]}`  \n"
    f"**duration:** {manifest.get('duration_s') or 0:.0f}s  \n"
    f"**python:** {manifest['environment']['python']}"
)

page = st.sidebar.radio(
    "View",
    ["Results", "Correction decisions", "Instance inspector", "Reliability signals",
     "Benchmark composition", "Reproducibility"],
)

scores = run.get("scores", pd.DataFrame())
signals = run.get("signals", pd.DataFrame())
labels = run.get("test_labels", pd.DataFrame())


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #

if page == "Results":
    st.header("Policy comparison")
    comparison = run.get("comparison")
    if comparison is None or comparison.empty:
        st.info("No comparison table in this run.")
    else:
        st.dataframe(
            comparison[
                ["policy", "mean_metric", "relative_change_pct", "ci_low", "ci_high",
                 "p_value", "p_adjusted", "significant", "cliffs_delta", "n_instances"]
            ].style.format(precision=4),
            use_container_width=True,
        )
        st.caption(
            "`relative_change_pct` is relative to the never-correcting reference; "
            "negative is better. `p_adjusted` is Holm-corrected across the hypothesis "
            "family. Confidence intervals are series-level bootstrap."
        )

    if not scores.empty:
        st.subheader("Headroom: achievable vs. apparent")
        realizable = run.get("realizable_oracle")
        cols = st.columns(4)
        accept = float(labels["accept_metric"].mean()) if not labels.empty else np.nan
        cols[0].metric("Accept (no correction)", f"{accept:.4f}")
        best = scores.groupby("policy")["chosen_metric"].mean().sort_values()
        cols[1].metric("Best policy", f"{best.iloc[0]:.4f}", best.index[0])
        if realizable is not None and not realizable.empty:
            cols[2].metric(
                "Realizable oracle", f"{realizable['realizable_metric'].mean():.4f}",
                help="Best action chosen on validation -- achievable, unbiased.",
            )
        if not labels.empty:
            cols[3].metric(
                "Unrestricted oracle", f"{labels['oracle_metric'].mean():.4f}",
                help=(
                    "Best of all actions chosen on test outcomes. Inflated by "
                    "selection over ~10 noisy candidates; an upper bound no policy "
                    "can attain."
                ),
            )

        st.subheader("Mean metric by policy")
        st.bar_chart(best)

elif page == "Correction decisions":
    st.header("What the corrections actually did")
    if scores.empty:
        st.info("No scores in this run.")
        st.stop()

    rows = []
    for policy, group in scores.groupby("policy"):
        corrected = group[group["corrected"]]
        rows.append(
            {
                "policy": policy,
                "correction rate": group["corrected"].mean(),
                "helped": corrected["improved"].mean() if len(corrected) else np.nan,
                "harmed": corrected["harmed"].mean() if len(corrected) else np.nan,
                "decision accuracy": group["decision_correct"].mean(),
                "mean regret": group["regret"].mean(),
                "mean MASE": group["chosen_metric"].mean(),
            }
        )
    st.dataframe(
        pd.DataFrame(rows).sort_values("mean regret").style.format(precision=4),
        use_container_width=True,
    )
    st.caption(
        "`helped` / `harmed` are computed among corrected instances only. A policy can "
        "correct often and still be worse than never correcting if `harmed` exceeds "
        "`helped`."
    )

    st.subheader("Chosen actions")
    decisions = run.get("decisions")
    if decisions is not None and not decisions.empty:
        policy = st.selectbox("Policy", sorted(decisions["policy"].unique()))
        sub = decisions[decisions["policy"] == policy]
        left, right = st.columns(2)
        left.bar_chart(sub["chosen_action"].value_counts())
        if "diagnosed_mode" in sub.columns:
            right.bar_chart(sub["diagnosed_mode"].value_counts())
        st.subheader("Reasoning traces")
        st.dataframe(
            sub[["instance_id", "chosen_action", "risk", "gain", "confidence",
                 "diagnosed_mode", "rationale"]].head(200),
            use_container_width=True,
        )

elif page == "Instance inspector":
    st.header("Instance inspector")
    if signals.empty:
        st.info("No signals in this run.")
        st.stop()

    synthetic = signals[signals["source"] == "synthetic"]
    if synthetic.empty:
        st.info("This run contains no synthetic series, which are the ones the "
                "inspector can regenerate exactly.")
        st.stop()

    family = st.selectbox("Family", sorted(synthetic["family"].unique()))
    subset = synthetic[synthetic["family"] == family]
    instance_id = st.selectbox("Instance", sorted(subset["instance_id"].unique()))
    row = subset[subset["instance_id"] == instance_id].iloc[0]

    cols = st.columns(4)
    cols[0].metric("Horizon", int(row["horizon"]))
    cols[1].metric("Base model", row["base_model"])
    cols[2].metric("Backtest MASE", f"{row.get('backtest.backtest_mase', float('nan')):.3f}")
    cols[3].metric("Disagreement", f"{row.get('disagreement.normalized', float('nan')):.3f}")

    st.subheader("Forecasts and diagnostics")
    with st.spinner("Recomputing this instance..."):
        try:
            import matplotlib.pyplot as plt

            from benchmark.actions import ActionContext, apply_all_actions
            from forecasting.base import FORECASTERS
            from tools.base import TOOLS
            from visualization.figures import set_style

            set_style()
            index = int(str(row["series_id"]).split("_")[-1])
            n_timesteps = int(row["context.history_length"]) + 200
            series = regenerate_series(family, index, n_timesteps)
            origin = int(row["origin"])
            horizon = int(row["horizon"])
            history = series.target[:origin]

            pool = {
                m: FORECASTERS.build(m).forecast(
                    history, horizon, series.seasonal_period, seed=0
                )
                for m in ("naive", "seasonal_naive", "theta", "drift")
            }
            diagnostics = {
                n: TOOLS.build(n).run(history, series.seasonal_period).values
                for n in ("changepoint", "anomaly", "seasonality", "stl")
            }
            ctx = ActionContext(
                history=history, seasonal_period=series.seasonal_period,
                horizon=horizon, base_model=str(row["base_model"]),
                base_result=pool.get(str(row["base_model"]), pool["theta"]),
                diagnostics=diagnostics, pool=pool,
            )
            actions = apply_all_actions(ctx, ["accept", "damp_trend", "ensemble", "abstain"])

            from datasets.base import Instance

            inst = Instance(instance_id, series.series_id, origin, horizon, "test")
            from visualization.figures import fig_example_forecast

            fig = fig_example_forecast(series, inst, actions, diagnostics)
            st.pyplot(fig)

            left, right = st.columns(2)
            left.write("**Detected change points**")
            left.write(diagnostics["changepoint"].get("changepoints") or "none")
            left.write("**True change points (ground truth)**")
            left.write(list(series.labels.changepoints) or "none")
            right.write("**Detected anomalies**")
            right.write(
                f"{diagnostics['anomaly'].get('n_anomalies', 0)} "
                f"({diagnostics['anomaly'].get('anomaly_rate', 0):.1%})"
            )
            right.write("**True anomalies (ground truth)**")
            right.write(len(series.labels.anomaly_indices))
        except Exception as exc:  # noqa: BLE001 - inspector must not break the app
            st.error(f"Could not reconstruct this instance: {type(exc).__name__}: {exc}")

elif page == "Reliability signals":
    st.header("Reliability signals")
    importance = run.get("feature_importance")
    if importance is not None and not importance.empty:
        st.subheader("What drives the correction gate")
        st.bar_chart(importance.head(15).set_index("feature")["importance"])
        st.caption("Permutation importance (drop in AUROC when the signal is shuffled).")

    if not signals.empty and not labels.empty:
        st.subheader("Signal distributions by outcome")
        numeric = [
            c for c in signals.columns
            if signals[c].dtype.kind in "fi" and not c.startswith("label_")
        ]
        signal = st.selectbox("Signal", sorted(numeric),
                              index=sorted(numeric).index("backtest.backtest_mase")
                              if "backtest.backtest_mase" in numeric else 0)
        merged = signals.merge(
            labels[["instance_id", "large_error", "should_correct"]], on="instance_id"
        )
        split_by = st.radio("Split by", ["large_error", "should_correct"],
                            horizontal=True)
        summary = merged.groupby(split_by)[signal].describe()
        st.dataframe(summary.style.format(precision=4), use_container_width=True)

elif page == "Benchmark composition":
    st.header("Benchmark composition")
    if signals.empty:
        st.info("No signals in this run.")
        st.stop()
    summary = (
        signals.groupby(["source", "family"])
        .agg(series=("series_id", "nunique"), instances=("instance_id", "count"),
             median_length=("context.history_length", "median"))
        .reset_index()
    )
    st.dataframe(summary, use_container_width=True)
    left, right = st.columns(2)
    left.bar_chart(signals["family"].value_counts())
    right.bar_chart(signals["horizon"].value_counts())

    if not labels.empty:
        st.subheader("Correctable instances by family")
        st.bar_chart(
            labels.groupby("family")["should_correct"].mean().sort_values(ascending=False)
        )

elif page == "Reproducibility":
    st.header("Run provenance")
    st.subheader("Environment")
    st.json(manifest["environment"])
    git = manifest["environment"].get("git", {})
    if git.get("available") and git.get("dirty"):
        st.warning(
            "The working tree was dirty when this run executed, so the recorded commit "
            "does not fully describe the code that produced these results."
        )
    st.subheader("Counters")
    st.json(manifest.get("counters", {}))
    st.subheader("Configuration")
    st.json(manifest["config"])
    st.caption(
        "Reproduce this run with the same config file and seed. LLM arms replay from "
        "the response cache, so no API key is required to regenerate published results."
    )
