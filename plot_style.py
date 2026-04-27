"""
Publication-ready plotting style for low-visibility paper evaluation.
Journal-ready: 300 dpi, colorblind-safe palettes, consistent typography.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

# Palette per plan: Fog=deep blue, Mist=amber/orange, Clear=neutral gray
PALETTE = {
    "Fog":   "#2E5A87",  # deep blue
    "Mist":  "#E69F00",  # amber
    "Clear": "#7F7F7F",  # neutral gray
}
CLASS_COLORS = [PALETTE["Fog"], PALETTE["Mist"], PALETTE["Clear"]]
CLASS_NAMES  = ["Fog (<500 m)", "Mist (500–1000 m)", "Clear (≥1000 m)"]
CLASS_SHORT  = ["Fog", "Mist", "Clear"]

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
        "font.size":        10,
        "axes.labelsize":   11,
        "axes.titlesize":   12,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "legend.fontsize":  9,
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
    """Set default color cycle for Fog/Mist/Clear."""
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
            fontsize=12, fontweight="bold", va="bottom")


def get_fog_mist_clear_colors():
    return CLASS_COLORS, CLASS_NAMES, CLASS_SHORT
