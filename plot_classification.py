"""
Paper Figure 3, 4, 5: Classification performance, PR curves, reliability.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

from .plot_style import setup_paper_style, apply_palette, save_figure, PALETTE
from .metrics_core import (
    threshold_sweep_metrics,
    compute_calibration_metrics,
)


def plot_confusion_matrix_normalized(
    y_true,
    y_pred,
    class_names,
    output_path,
    ax=None,
    baseline_pred=None,
    baseline_name="IFS",
    model_name="PMST",
):
    """
    Normalized confusion matrix (rows=True).
    If baseline_pred is provided, draw two panels: (a) model, (b) baseline (e.g. IFS).
    """
    from sklearn.metrics import confusion_matrix

    def _draw_cm(ax, y_true, y_pred, title):
        # Restrict to valid predictions only (e.g. IFS may have -1 for unmatched)
        valid = (y_pred >= 0) & (y_pred < len(class_names))
        y_t = np.asarray(y_true)[valid]
        y_p = np.asarray(y_pred)[valid]
        if len(y_t) == 0:
            ax.text(0.5, 0.5, "No valid samples", transform=ax.transAxes, ha="center", va="center")
            return
        cm = confusion_matrix(y_t, y_p, labels=np.arange(len(class_names)))
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                txt = f"{cm_norm[i, j]:.2f}\n(n={cm[i, j]})"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="Fraction")

    setup_paper_style()
    if baseline_pred is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        _draw_cm(axes[0], y_true, y_pred, model_name)
        valid = (np.asarray(baseline_pred) >= 0) & (np.asarray(baseline_pred) < len(class_names))
        _draw_cm(axes[1], y_true, baseline_pred, baseline_name)
        plt.tight_layout()
        if output_path:
            save_figure(fig, output_path)
        return fig, axes
    else:
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        else:
            fig = ax.get_figure()
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                txt = f"{cm_norm[i, j]:.2f}\n(n={cm[i, j]})"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.colorbar(im, ax=ax, label="Fraction")
        plt.tight_layout()
        if output_path:
            save_figure(fig, output_path)
        return fig, ax


def _f1_from_pr(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def plot_per_class_prf1(
    stats,
    output_path,
    baseline_stats=None,
    baseline_name="IFS",
    model_name="PMST",
):
    """
    Per-class P/R/F1 bar chart.
    If baseline_stats is provided, draw grouped bars: PMST vs baseline (e.g. IFS) for each metric.
    """
    setup_paper_style()
    apply_palette()
    class_names = ["Ultra-low", "Moderate-low", "Clear"]
    metrics = ["Precision", "Recall", "F1"]
    keys_p = ["Fog_P", "Mist_P", "Clear_P"]
    keys_r = ["Fog_R", "Mist_R", "Clear_R"]
    keys_f1 = ["Fog_F1", "Mist_F1", "Clear_F1"]

    def _get_vals(s):
        prec = [s.get(k, 0) for k in keys_p]
        rec = [s.get(k, 0) for k in keys_r]
        f1 = [
            _f1_from_pr(p, r) if s.get(keys_f1[i]) is None else s.get(keys_f1[i], 0)
            for i, (p, r) in enumerate(zip(prec, rec))
        ]
        return prec, rec, f1

    if baseline_stats is None:
        prec, rec, f1 = _get_vals(stats)
        x = np.arange(len(class_names))
        w = 0.25
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(x - w, prec, w, label="Precision")
        ax.bar(x, rec, w, label="Recall")
        ax.bar(x + w, f1, w, label="F1")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.set_ylabel("Score")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        if output_path:
            save_figure(fig, output_path)
        return fig

    # Two models: grouped bars per class, three metrics each
    pmst_prec, pmst_rec, pmst_f1 = _get_vals(stats)
    base_prec, base_rec, base_f1 = _get_vals(baseline_stats)
    x = np.arange(len(class_names))
    width = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (mname, pmst_vals, base_vals) in zip(
        axes,
        [
            ("Precision", pmst_prec, base_prec),
            ("Recall", pmst_rec, base_rec),
            ("F1", pmst_f1, base_f1),
        ],
    ):
        bars1 = ax.bar(x - width / 2, pmst_vals, width, label=model_name, color=PALETTE["Fog"])
        bars2 = ax.bar(x + width / 2, base_vals, width, label=baseline_name, color="#5B5B5B")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.set_ylabel("Score")
        ax.set_title(mname)
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_pr_curves(probs, y_true, class_names_short, output_path):
    """Ultra-low and Moderate-low one-vs-rest PR curves."""
    setup_paper_style()
    apply_palette()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for idx, (cls_name, ax) in enumerate(zip(class_names_short[:2], axes)):
        binary = (y_true == idx).astype(int)
        prec, rec, _ = precision_recall_curve(binary, probs[:, idx])
        ap = average_precision_score(binary, probs[:, idx])
        ax.plot(rec, prec, lw=2, label=f"{cls_name} (AP={ap:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_threshold_sweep(probs, y_true, output_path, fog_th=0.46, mist_th=0.38):
    """POD vs FAR and CSI vs threshold sweep."""
    setup_paper_style()
    apply_palette()
    sweep = threshold_sweep_metrics(probs, y_true, np.linspace(0.1, 0.9, 41))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Left: POD vs FAR
    ax = axes[0]
    ax.plot(sweep["far_fog"], sweep["pod_fog"], "C0-", lw=2, label="Ultra-low POD vs FAR")
    ax.plot(sweep["far_mist"], sweep["pod_mist"], "C1-", lw=2, label="Moderate-low POD vs FAR")
    ax.axvline(x=0.5, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("False Alarm Ratio")
    ax.set_ylabel("Probability of Detection")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    # Right: CSI vs threshold
    ax = axes[1]
    ax.plot(sweep["fog_thresholds"], sweep["csi_fog"], "C0-", lw=2, label="Ultra-low CSI")
    ax.plot(sweep["mist_thresholds"], sweep["csi_mist"], "C1-", lw=2, label="Moderate-low CSI")
    ax.axvline(x=fog_th, color="C0", ls="--", alpha=0.7, label=f"Ultra-low op={fog_th}")
    ax.axvline(x=mist_th, color="C1", ls="--", alpha=0.7, label=f"Moderate-low op={mist_th}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("CSI")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_reliability_diagram(probs, y_true, output_path, n_bins=10):
    """Reliability diagram for Ultra-low and Moderate-low probabilities."""
    setup_paper_style()
    apply_palette()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ece, brier, bin_acc, bin_conf = compute_calibration_metrics(
        probs, y_true, n_bins=n_bins
    )
    for idx, (cls_name, ax) in enumerate(zip(["Ultra-low", "Moderate-low"], axes)):
        acc = bin_acc[idx]
        conf = bin_conf[idx]
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.bar(conf - 0.5 / n_bins, acc, width=0.9 / n_bins, alpha=0.7, label=cls_name)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"{cls_name} (ECE={ece[idx]:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    plt.suptitle(f"Brier: Ultra-low={brier[0]:.3f}, Moderate-low={brier[1]:.3f}")
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig
