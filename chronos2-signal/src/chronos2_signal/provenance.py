"""Hashing, cache keys and environment fingerprints.

A research result is only meaningful together with the data snapshot, the code,
the model weights, the policy and the cost assumptions that produced it. This
module produces the identifiers that bind those together, so a stored forecast
can never be silently re-attributed to a different context.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "sha256_bytes",
    "sha256_file",
    "stable_hash",
    "forecast_cache_key",
    "EnvironmentFingerprint",
    "capture_environment",
    "ReleaseManifest",
]

#: Packages whose versions materially change numerical output.
_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "pandas-market-calendars",
    "pyarrow",
    "PyYAML",
    "yfinance",
    "chronos-forecasting",
    "torch",
    "transformers",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Stream a file through SHA256.

    Used to verify a downloaded checkpoint against its recorded hash rather
    than trusting that a revision tag resolved to the expected bytes.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(obj: Any) -> str:
    """SHA256 over a canonical JSON rendering of ``obj``."""
    payload = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return sha256_bytes(payload.encode("utf-8"))


def forecast_cache_key(
    *,
    checkpoint_revision: str,
    package_version: str,
    schema_version: int,
    symbol: str,
    origin_session: str,
    context_length: int,
    prediction_length: int,
    source_snapshot_hash: str,
    channel_set: Sequence[str],
    cross_learning: bool,
) -> str:
    """Identifier for one cached causal forecast.

    Every input that can change the numbers is part of the key. In particular
    the channel set and the cross-learning flag are included: sharing across
    tasks makes a prediction depend on batch composition, so a cached value
    produced under a different setting is a different value.
    """
    return stable_hash(
        {
            "checkpoint_revision": checkpoint_revision,
            "package_version": package_version,
            "schema_version": schema_version,
            "symbol": symbol,
            "origin_session": origin_session,
            "context_length": context_length,
            "prediction_length": prediction_length,
            "source_snapshot_hash": source_snapshot_hash,
            "channel_set": list(channel_set),
            "cross_learning": cross_learning,
        }
    )


@dataclass(frozen=True)
class EnvironmentFingerprint:
    """Installed versions and hardware, captured before results are examined."""

    python_version: str
    platform: str
    packages: Mapping[str, str]
    device: str
    torch_version: str | None = None
    cuda_available: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "packages": dict(self.packages),
            "device": self.device,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
        }

    def digest(self) -> str:
        return stable_hash(self.to_dict())

    def missing_packages(self) -> list[str]:
        return sorted(name for name, version in self.packages.items() if version == "absent")


def capture_environment() -> EnvironmentFingerprint:
    """Record the current environment.

    Versions are read from installed metadata rather than written by hand: a
    hand-written version string in a report is an unverified claim.
    """
    packages: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = "absent"

    torch_version: str | None = None
    cuda_available: bool | None = None
    device = "cpu"
    try:  # pragma: no cover - depends on the optional model extra
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        device = "cuda" if cuda_available else "cpu"
    except Exception:
        torch_version = None
        cuda_available = None

    return EnvironmentFingerprint(
        python_version=sys.version,
        platform=platform.platform(),
        packages=packages,
        device=device,
        torch_version=torch_version,
        cuda_available=cuda_available,
    )


@dataclass(frozen=True)
class ReleaseManifest:
    """Binds data, code, weights, policy and costs into one identity.

    Written next to any result that is reported. A result without a manifest is
    not reportable, because there is no way to say later what produced it.
    """

    design_version: str
    design_fingerprint: str
    code_revision: str
    checkpoint_id: str
    checkpoint_revision: str
    checkpoint_sha256: str | None
    calendar: Mapping[str, str]
    environment: EnvironmentFingerprint
    cost_scenarios: Mapping[str, float]
    universe_manifest_hash: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "design_version": self.design_version,
            "design_fingerprint": self.design_fingerprint,
            "code_revision": self.code_revision,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_revision": self.checkpoint_revision,
            "checkpoint_sha256": self.checkpoint_sha256,
            "calendar": dict(self.calendar),
            "environment": self.environment.to_dict(),
            "cost_scenarios": dict(self.cost_scenarios),
            "universe_manifest_hash": self.universe_manifest_hash,
            "notes": list(self.notes),
        }

    def digest(self) -> str:
        return stable_hash(self.to_dict())

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.to_dict())
        payload["manifest_digest"] = self.digest()
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target
