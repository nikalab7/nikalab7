"""Typed access to the registered design configuration.

The YAML file in ``config/design.yaml`` is the single source of truth for every
threshold in the protocol. Nothing in this package is allowed to invent a
threshold: modules read them from here so that a protocol amendment is a
one-line diff with a visible fingerprint.

The loader is deliberately strict. An unknown key, a missing key or a value
outside its declared domain raises :class:`ConfigError` instead of silently
falling back to a default, because a silent default is how a research protocol
drifts away from the thing that was preregistered.
"""

from __future__ import annotations

import datetime as dt
import functools
import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, get_args, get_origin, get_type_hints

import yaml

__all__ = [
    "ConfigError",
    "VALIDATED_STATUSES",
    "KNOWN_STATUSES",
    "DataConfig",
    "UniverseConfig",
    "ModelConfig",
    "DecisionConfig",
    "ExecutionConfig",
    "PortfolioConfig",
    "ValidationConfig",
    "EventsConfig",
    "RuntimeConfig",
    "DesignConfig",
    "load_design",
    "default_config_path",
]


class ConfigError(ValueError):
    """Raised when the registered configuration is missing, extra or invalid."""


#: Statuses that mean the registered evaluation gates have been passed. Only a
#: status in this set unlocks the ``QUALIFIED SIGNAL`` label.
VALIDATED_STATUSES = frozenset({"validated", "qualified_signal"})

#: Statuses the loader accepts. A typo is rejected rather than silently
#: reinterpreted, because the status decides the output label.
KNOWN_STATUSES = frozenset(
    {"design_only_unvalidated", "unvalidated", "forward_paper", *VALIDATED_STATUSES}
)


@dataclass(frozen=True)
class DataConfig:
    provider: str
    hourly_interval: str
    hourly_request_lookback_days: int
    daily_request_years: int
    archive_interval: str
    archive_initial_lookback_days: int
    prepost: bool
    auto_adjust: bool
    actions: bool
    automatic_repair: bool
    snapshot_policy: str
    exchange_timezone: str
    storage_timezone: str
    max_parallel_requests: int
    max_retries: int


@dataclass(frozen=True)
class UniverseConfig:
    max_issuers: int
    max_issuers_per_sector: int
    candidate_manifest_required: bool
    min_asof_price_usd: float
    min_median_daily_dollar_volume_usd: float
    liquidity_window_sessions: int
    min_daily_observations: int
    common_hourly_history_bars: int
    min_observed_fraction: float
    latest_session_complete_required: bool


@dataclass(frozen=True)
class ModelConfig:
    id: str
    revision: str
    context_length: int
    context_candidates: tuple[int, ...]
    horizon_sessions: int
    prediction_length_policy: str
    quantile_levels: tuple[float, ...]
    target: str
    cross_learning: bool
    batch_size_channels: int
    dtype: str
    finetune: bool
    future_inputs: str
    safetensors_sha256: str


@dataclass(frozen=True)
class DecisionConfig:
    return_estimator: str
    ridge_alpha: float
    probability_estimator: str
    logistic_C: float
    logistic_max_iter: int
    calibration: str
    date_total_sample_weight: float
    standardized_feature_clip_abs: float
    min_probability: float
    min_estimated_net_return: float
    require_positive_stress_estimate: bool
    ranking: str
    sigma_2d_floor: float
    sigma_daily_window_sessions: int


@dataclass(frozen=True)
class ExecutionConfig:
    signal_time: str
    max_data_delay_after_close_minutes: int
    entry: str
    exit: str
    base_slippage_bps_per_side: float
    stress_slippage_bps_per_side: float
    severe_slippage_bps_per_side: float
    explicit_broker_fees: str
    explicit_fee_fraction: float
    gap_filter: str
    stop_loss: str
    take_profit: str
    automatic_orders: bool

    def slippage_fraction(self, scenario: str) -> float:
        """Per-side slippage as a fraction for a named cost scenario."""
        table = {
            "base": self.base_slippage_bps_per_side,
            "stress": self.stress_slippage_bps_per_side,
            "severe": self.severe_slippage_bps_per_side,
        }
        if scenario not in table:
            raise ConfigError(
                f"unknown cost scenario {scenario!r}; expected one of {sorted(table)}"
            )
        return table[scenario] / 10_000.0


@dataclass(frozen=True)
class PortfolioConfig:
    max_new_alerts_per_origin: int
    max_open_positions: int
    max_open_per_sector: int
    pairwise_correlation_cap: float
    correlation_window_sessions: int
    position_fraction_current_equity_at_entry: float
    max_gross_exposure: float
    pause_new_alerts_at_portfolio_drawdown: float
    initial_equity: float


@dataclass(frozen=True)
class ValidationConfig:
    earliest_primary_origin: dt.date
    final_historical_test_origin_sessions: int
    min_fit_origin_sessions: int
    calibration_origin_sessions: int
    validation_origin_sessions: int
    purge_sessions_per_boundary: int
    require_label_available_timestamp_check: bool
    decision_refit_every_sessions: int
    initial_learned_variants: tuple[str, ...]
    bootstrap_samples: int
    bootstrap_block_sessions: int
    bootstrap_block_sensitivity: tuple[int, ...]
    forward_min_sessions: int
    forward_min_executed_signals: int
    combined_evaluation_min_executed_signals: int
    combined_evaluation_min_signal_sessions: int
    development_min_executed_trades: int
    development_min_signal_sessions: int
    development_selection_bootstrap_confidence: float
    promotion_bootstrap_confidence: float
    promotion_min_profitable_blocks: int
    promotion_block_sessions: int


@dataclass(frozen=True)
class EventsConfig:
    version1_mode: str
    future_event_veto_requires_new_version: bool
    missing_event_status: str


@dataclass(frozen=True)
class RuntimeConfig:
    numerical_store: str
    ledger: str
    device: str
    external_paid_services_required: bool
    exact_dependency_lock_required_before_run: bool
    schema_version: int
    exchange_calendar: str
    market_proxy: str


@dataclass(frozen=True)
class DesignConfig:
    """The whole registered protocol, as loaded from disk."""

    design_version: str
    status: str
    data: DataConfig
    universe: UniverseConfig
    model: ModelConfig
    decision: DecisionConfig
    execution: ExecutionConfig
    portfolio: PortfolioConfig
    validation: ValidationConfig
    events: EventsConfig
    runtime: RuntimeConfig
    source_path: Path | None = field(default=None, compare=False)

    @property
    def is_validated(self) -> bool:
        """Whether the protocol has passed its evaluation gates.

        Deliberately an allowlist rather than a denylist: only a status that
        *explicitly* records a passed evaluation counts as validated. An
        unrecognised, misspelled or experimental status reads as unvalidated, so
        the output label fails closed to ``RESEARCH / UNVALIDATED`` instead of
        promoting itself by accident.
        """
        return self.status in VALIDATED_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return _as_plain(self)

    def fingerprint(self) -> str:
        """Stable SHA256 over the effective configuration.

        Recorded in fit and release manifests so that a result can never be
        silently re-attributed to a different set of thresholds.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "universe": UniverseConfig,
    "model": ModelConfig,
    "decision": DecisionConfig,
    "execution": ExecutionConfig,
    "portfolio": PortfolioConfig,
    "validation": ValidationConfig,
    "events": EventsConfig,
    "runtime": RuntimeConfig,
}


def default_config_path() -> Path:
    """Path to the registered configuration shipped with the package."""
    return Path(__file__).resolve().parents[2] / "config" / "design.yaml"


def load_design(path: str | Path | None = None) -> DesignConfig:
    """Load and validate the registered design configuration."""
    config_path = Path(path) if path is not None else default_config_path()
    if not config_path.is_file():
        raise ConfigError(f"design configuration not found at {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{config_path} must contain a YAML mapping")

    missing = {"design_version", "status", *_SECTIONS} - set(raw)
    if missing:
        raise ConfigError(f"design configuration missing keys: {sorted(missing)}")
    extra = set(raw) - {"design_version", "status", *_SECTIONS}
    if extra:
        raise ConfigError(f"design configuration has unknown keys: {sorted(extra)}")

    sections: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        block = raw[name]
        if not isinstance(block, Mapping):
            raise ConfigError(f"section {name!r} must be a mapping")
        sections[name] = _build(cls, block, name)

    config = DesignConfig(
        design_version=str(raw["design_version"]),
        status=str(raw["status"]),
        source_path=config_path,
        **sections,
    )
    _validate(config)
    return config


@functools.lru_cache(maxsize=None)
def _declared_types(cls: type) -> dict[str, Any]:
    """Resolved annotations for a config dataclass.

    ``from __future__ import annotations`` leaves ``Field.type`` as a string, so
    the annotations are resolved against this module's namespace instead.
    """
    hints = get_type_hints(cls)
    return {f.name: hints[f.name] for f in fields(cls)}


def _build(cls: type, block: Mapping[str, Any], section: str) -> Any:
    declared = _declared_types(cls)
    unknown = set(block) - set(declared)
    if unknown:
        raise ConfigError(f"section {section!r} has unknown keys: {sorted(unknown)}")
    missing = set(declared) - set(block)
    if missing:
        raise ConfigError(f"section {section!r} missing keys: {sorted(missing)}")
    kwargs = {
        name: _coerce(block[name], annotation, f"{section}.{name}")
        for name, annotation in declared.items()
    }
    return cls(**kwargs)


def _coerce(value: Any, annotation: Any, where: str) -> Any:
    origin = get_origin(annotation)
    if origin is tuple:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ConfigError(f"{where} must be a list")
        (item_type, _ellipsis) = get_args(annotation)
        return tuple(_coerce(item, item_type, where) for item in value)
    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where} must be a boolean, got {value!r}")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where} must be an integer, got {value!r}")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where} must be a number, got {value!r}")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where} must be a string, got {value!r}")
        return value
    if annotation is dt.date:
        if isinstance(value, dt.datetime):
            raise ConfigError(f"{where} must be a plain date, not a datetime")
        if isinstance(value, dt.date):
            return value
        if isinstance(value, str):
            return dt.date.fromisoformat(value)
        raise ConfigError(f"{where} must be a date, got {value!r}")
    raise ConfigError(f"{where} has unsupported annotation {annotation!r}")


def _validate(config: DesignConfig) -> None:
    """Domain checks that a YAML type cannot express."""
    if config.status not in KNOWN_STATUSES:
        raise ConfigError(
            f"status {config.status!r} is not recognised; expected one of "
            f"{sorted(KNOWN_STATUSES)}. The status decides the output label, so a "
            "typo is rejected rather than reinterpreted."
        )
    model, decision, universe = config.model, config.decision, config.universe
    execution, portfolio, validation = config.execution, config.portfolio, config.validation
    data = config.data

    quantiles = model.quantile_levels
    if list(quantiles) != sorted(quantiles) or len(set(quantiles)) != len(quantiles):
        raise ConfigError("model.quantile_levels must be strictly increasing")
    if not all(0.0 < q < 1.0 for q in quantiles):
        raise ConfigError("model.quantile_levels must lie strictly inside (0, 1)")
    if 0.5 not in quantiles:
        raise ConfigError("model.quantile_levels must include the 0.50 median level")
    if model.context_length not in model.context_candidates:
        raise ConfigError("model.context_length must be one of model.context_candidates")
    if model.context_length > 8192:
        raise ConfigError("model.context_length exceeds the checkpoint maximum of 8192")
    if model.horizon_sessions < 1:
        raise ConfigError("model.horizon_sessions must be at least 1")
    if model.batch_size_channels < 1:
        raise ConfigError("model.batch_size_channels must be positive")
    if model.finetune:
        raise ConfigError("model.finetune must stay false in design version 1")
    if model.cross_learning:
        raise ConfigError(
            "model.cross_learning must stay false: sharing across stock tasks makes "
            "predictions depend on batch composition"
        )
    if model.future_inputs != "calendar_only":
        raise ConfigError("model.future_inputs must be 'calendar_only'")
    if len(model.revision) != 40 or not all(c in "0123456789abcdef" for c in model.revision):
        raise ConfigError("model.revision must be a full 40-character git commit hash")

    if universe.max_issuers < 1:
        raise ConfigError("universe.max_issuers must be positive")
    if universe.max_issuers_per_sector < 1:
        raise ConfigError("universe.max_issuers_per_sector must be positive")
    if not 0.0 < universe.min_observed_fraction <= 1.0:
        raise ConfigError("universe.min_observed_fraction must lie in (0, 1]")
    if universe.common_hourly_history_bars < model.context_length:
        raise ConfigError(
            "universe.common_hourly_history_bars must cover the longest context studied"
        )
    if universe.common_hourly_history_bars < max(model.context_candidates):
        raise ConfigError(
            "universe.common_hourly_history_bars must cover every context candidate so "
            "that all variants share one eligibility mask"
        )

    if not 0.0 < decision.min_probability < 1.0:
        raise ConfigError("decision.min_probability must lie strictly inside (0, 1)")
    if decision.min_estimated_net_return <= 0.0:
        raise ConfigError("decision.min_estimated_net_return must be positive")
    if decision.ridge_alpha <= 0.0 or decision.logistic_C <= 0.0:
        raise ConfigError("regularisation strengths must be positive")
    if decision.standardized_feature_clip_abs <= 0.0:
        raise ConfigError("decision.standardized_feature_clip_abs must be positive")
    if decision.sigma_2d_floor <= 0.0:
        raise ConfigError("decision.sigma_2d_floor must be positive")
    if decision.calibration != "sigmoid_on_later_disjoint_dates":
        raise ConfigError(
            "decision.calibration must be sigmoid on later disjoint dates; random-split "
            "calibration leaks across the time boundary"
        )

    if not (
        execution.base_slippage_bps_per_side
        <= execution.stress_slippage_bps_per_side
        <= execution.severe_slippage_bps_per_side
    ):
        raise ConfigError("slippage scenarios must be ordered base <= stress <= severe")
    if execution.base_slippage_bps_per_side < 0.0:
        raise ConfigError("slippage assumptions must be non-negative")
    if execution.explicit_fee_fraction < 0.0:
        raise ConfigError("execution.explicit_fee_fraction must be non-negative")
    if execution.automatic_orders:
        raise ConfigError("execution.automatic_orders must stay false: no broker in scope")
    for disabled in ("gap_filter", "stop_loss", "take_profit"):
        if getattr(execution, disabled) != "disabled":
            raise ConfigError(
                f"execution.{disabled} must be 'disabled' in version 1; enabling it "
                "changes the labels and needs a new registered execution experiment"
            )
    if execution.max_data_delay_after_close_minutes < 30:
        raise ConfigError(
            "execution.max_data_delay_after_close_minutes must not precede the "
            "close-plus-30-minutes signal time"
        )

    if portfolio.max_new_alerts_per_origin < 1:
        raise ConfigError("portfolio.max_new_alerts_per_origin must be positive")
    if portfolio.max_open_positions < 1:
        raise ConfigError("portfolio.max_open_positions must be positive")
    if not 0.0 < portfolio.pairwise_correlation_cap <= 1.0:
        raise ConfigError("portfolio.pairwise_correlation_cap must lie in (0, 1]")
    if not 0.0 < portfolio.position_fraction_current_equity_at_entry <= 1.0:
        raise ConfigError("portfolio.position_fraction_current_equity_at_entry must lie in (0, 1]")
    if not 0.0 < portfolio.max_gross_exposure <= 1.0:
        raise ConfigError(
            "portfolio.max_gross_exposure must lie in (0, 1]: the reference portfolio "
            "does not lever"
        )
    if not 0.0 < portfolio.pause_new_alerts_at_portfolio_drawdown < 1.0:
        raise ConfigError("portfolio.pause_new_alerts_at_portfolio_drawdown must lie in (0, 1)")
    if portfolio.initial_equity <= 0.0:
        raise ConfigError("portfolio.initial_equity must be positive")

    for name in (
        "final_historical_test_origin_sessions",
        "min_fit_origin_sessions",
        "calibration_origin_sessions",
        "validation_origin_sessions",
        "decision_refit_every_sessions",
        "bootstrap_samples",
        "bootstrap_block_sessions",
        "promotion_block_sessions",
    ):
        if getattr(validation, name) < 1:
            raise ConfigError(f"validation.{name} must be positive")
    if validation.purge_sessions_per_boundary < 1:
        raise ConfigError(
            "validation.purge_sessions_per_boundary must be at least 1: a two-session "
            "hold overlaps the following origins"
        )
    if validation.purge_sessions_per_boundary < model.horizon_sessions:
        raise ConfigError(
            "validation.purge_sessions_per_boundary must be at least the holding period "
            "in sessions so that outcome windows cannot straddle a split boundary"
        )
    if not validation.require_label_available_timestamp_check:
        raise ConfigError(
            "validation.require_label_available_timestamp_check must stay true: the "
            "timestamp check overrides the fixed gap length"
        )
    if set(validation.initial_learned_variants) != {"B0", "C128", "C256", "C512", "U256"}:
        raise ConfigError(
            "validation.initial_learned_variants must be exactly the five registered "
            "variants; a new variant needs a new design version"
        )
    if not 0.0 < validation.development_selection_bootstrap_confidence < 1.0:
        raise ConfigError("development selection confidence must lie in (0, 1)")
    if not 0.0 < validation.promotion_bootstrap_confidence < 1.0:
        raise ConfigError("promotion confidence must lie in (0, 1)")

    if data.max_parallel_requests < 1:
        raise ConfigError("data.max_parallel_requests must be positive")
    if data.max_retries < 0:
        raise ConfigError("data.max_retries must be non-negative")
    if data.prepost:
        raise ConfigError("data.prepost must stay false: the target is the regular session")
    if data.auto_adjust:
        raise ConfigError(
            "data.auto_adjust must stay false: execution prices and dividend-adjusted "
            "closes are stored separately"
        )
    if not data.actions:
        raise ConfigError("data.actions must stay true: action provenance is required")
    if data.automatic_repair:
        raise ConfigError(
            "data.automatic_repair must stay false: provider repairs would mutate the "
            "immutable snapshot"
        )
    if data.hourly_request_lookback_days > 730:
        raise ConfigError(
            "data.hourly_request_lookback_days must stay inside the provider's "
            "~730 calendar-day hourly window"
        )
    if data.archive_initial_lookback_days > 60:
        raise ConfigError(
            "data.archive_initial_lookback_days must stay inside the provider's "
            "~60 calendar-day 15-minute window"
        )

    if config.events.version1_mode != "annotations_only":
        raise ConfigError(
            "events.version1_mode must be 'annotations_only'; an event veto is a "
            "separately versioned policy tested prospectively"
        )
    if not config.events.future_event_veto_requires_new_version:
        raise ConfigError("events.future_event_veto_requires_new_version must stay true")
    if config.runtime.external_paid_services_required:
        raise ConfigError("runtime.external_paid_services_required must stay false")


def _as_plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _as_plain(getattr(value, f.name))
            for f in fields(value)
            if f.compare
        }
    if isinstance(value, tuple):
        return [_as_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dt.date):
        return value.isoformat()
    return value
