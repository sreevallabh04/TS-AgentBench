"""A tiny component registry.

Configs name components by string (``model: arima``); the registry turns that string
into a constructor. This keeps the experiment runner free of long if/elif chains and
means a new forecaster or corrective action becomes available to every experiment the
moment it is registered.
"""

from __future__ import annotations

from typing import Callable, Generic, Iterator, TypeVar

T = TypeVar("T")

__all__ = ["Registry", "RegistryError"]


class RegistryError(KeyError):
    """Raised when a component name is missing or double-registered."""


class Registry(Generic[T]):
    """Name -> factory mapping with decorator registration.

    >>> models: Registry[object] = Registry("model")
    >>> @models.register("naive")
    ... class Naive:
    ...     pass
    >>> models.get("naive") is Naive
    True
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, name: str, obj: Callable[..., T] | None = None):
        """Register ``obj`` under ``name``. Usable as a decorator or directly."""

        def _do(target: Callable[..., T]) -> Callable[..., T]:
            key = name.lower()
            if key in self._items and self._items[key] is not target:
                raise RegistryError(
                    f"{self.kind} {name!r} is already registered to "
                    f"{self._items[key]!r}; refusing to shadow it silently."
                )
            self._items[key] = target
            return target

        return _do if obj is None else _do(obj)

    def get(self, name: str) -> Callable[..., T]:
        key = name.lower()
        if key not in self._items:
            raise RegistryError(
                f"unknown {self.kind} {name!r}. Registered: {sorted(self._items)}"
            )
        return self._items[key]

    def build(self, name: str, **kwargs) -> T:
        """Construct a component by name."""
        return self.get(name)(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.lower() in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._items))

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Registry({self.kind!r}, {len(self._items)} items)"
