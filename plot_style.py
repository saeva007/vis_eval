"""
Publication-ready plotting style for low-visibility paper evaluation.
Journal-ready: 300 dpi, colorblind-safe palettes, consistent typography.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

# Internal keys keep the historical class names; display labels use threshold bins.
PALETTE = {
    "Fog":   "#B2182B",  # high-salience red for ultra-low visibility
    "Mist":  "#F4A582",  # lighter warm tone for moderate-low visibility
    "Clear": "#F7F7F7",  # near-white neutral for clear conditions
}
CLASS_COLORS = [PALETTE["Fog"], PALETTE["Mist"], PALETTE["Clear"]]
CLASS_NAMES  = ["Ultra-low (<500 m)", "Moderate-low (500–1000 m)", "Clear (≥1000 m)"]
CLASS_SHORT  = ["Ultra-low", "Moderate-low", "Clear"]

# Skill heatmaps
CMAP_SKILL   = "cividis"
CMAP_DIVERGING = "RdBu_r"


def setup_paper_style(font_family="DejaVu Serif"):
    """Alias for setup_style."""
    setup_style(font_family)


def setup_style(font_family="DejaVu Serif"):
    """Apply journal-ready rcParams."""
    plt.rcParams.update({
        "font.family":      font_family,
        "font.size":        12,
        "axes.labelsize":   13,
        "axes.titlesize":   14,
        "xtick.labelsize":  11,
        "ytick.labelsize":  11,
        "legend.fontsize":  11,
        "figure.dpi":       150,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "axes.grid":        True,
        "grid.alpha":      0.3,
        "axes.axisbelow":   True,
        "xtick.direction":  "out",
        "ytick.direction":  "out",
    })


def save_figure(fig, path, dpi=300):
    """Save figure to path. Creates parent dirs if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  [Fig] Saved → {path}")


def apply_palette():
    """Set default color cycle for Ultra-low/Moderate-low/Clear."""
    import matplotlib.pyplot as plt
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CLASS_COLORS)


def export_fig(fig, path, formats=("png",), dpi=300):
    """Save figure to path(s). Creates parent dirs if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    base, _ = os.path.splitext(path)
    for fmt in formats:
        out = f"{base}.{fmt}" if fmt != "png" or not path.endswith(".png") else path
        if not out.endswith(f".{fmt}"):
            out = f"{base}.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"  [Fig] Saved → {out}")


def add_panel_label(ax, label, x=-0.12, y=1.02):
    """Add panel letter (a), (b), (c) etc."""
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="bottom")


def get_fog_mist_clear_colors():
    return CLASS_COLORS, CLASS_NAMES, CLASS_SHORT
