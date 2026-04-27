# Paper evaluation package for PMST low-visibility model
import importlib
import numpy as np

_metrics_core = importlib.import_module(".metrics_core", __name__)
_plot_scenarios = importlib.import_module(".plot_scenarios", __name__)


def _fallback_pred_from_season_thresholds(
    probs, months, season_thresholds, default_fog_th=0.46, default_mist_th=0.38
):
    """
    Backward-compatible fallback for stale notebook/kernel module caches.
    This mirrors metrics_core.pred_from_season_thresholds.
    """
    month_to_season = {
        12: "DJF", 1: "DJF", 2: "DJF",
        3: "MAM", 4: "MAM", 5: "MAM",
        6: "JJA", 7: "JJA", 8: "JJA",
        9: "SON", 10: "SON", 11: "SON",
    }
    months = np.asarray(months, dtype=np.int32).ravel()
    n = probs.shape[0]
    fog_th = np.full(n, default_fog_th, dtype=np.float64)
    mist_th = np.full(n, default_mist_th, dtype=np.float64)
    for i in range(n):
        season = month_to_season.get(int(months[i]))
        if season and season in season_thresholds:
            fog_th[i] = season_thresholds[season]["fog_th"]
            mist_th[i] = season_thresholds[season]["mist_th"]
    return _metrics_core.pred_from_thresholds(probs, fog_th, mist_th)


# Patch stale module objects in long-lived notebook kernels.
if not hasattr(_metrics_core, "pred_from_season_thresholds"):
    _metrics_core.pred_from_season_thresholds = _fallback_pred_from_season_thresholds

compute_calibration_metrics = _metrics_core.compute_calibration_metrics
compute_rare_event_report = _metrics_core.compute_rare_event_report
threshold_sweep_metrics = _metrics_core.threshold_sweep_metrics
pred_from_thresholds = _metrics_core.pred_from_thresholds
pred_from_season_thresholds = _metrics_core.pred_from_season_thresholds
derive_scenario_columns = _plot_scenarios.derive_scenario_columns
plot_scenario_bars = _plot_scenarios.plot_scenario_bars
build_confusion_summaries_and_bottleneck_table = (
    _plot_scenarios.build_confusion_summaries_and_bottleneck_table
)

__all__ = [
    "compute_calibration_metrics",
    "compute_rare_event_report",
    "threshold_sweep_metrics",
    "pred_from_thresholds",
    "pred_from_season_thresholds",
    "derive_scenario_columns",
    "plot_scenario_bars",
    "build_confusion_summaries_and_bottleneck_table",
]
