"""Generate all assets: static plots + animation frame sequences."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nsga2_core import (
    NSGAConfig,
    assign_rank_and_distance,
    convergence_metric,
    diversity_metric,
    fast_non_dominated_sort,
    get_first_front,
    nsga2,
    sch,
    true_front_sch,
    true_front_zdt1,
    true_front_zdt2,
    true_front_zdt3,
    zdt1,
    zdt2,
    zdt3,
)

ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.dpi": 100})
CMAP = "RdYlGn_r"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rank_colors(F, fronts):
    """Return per-individual color based on front rank."""
    colors = np.zeros(len(F))
    for rank, front in enumerate(fronts):
        for i in front:
            colors[i] = rank
    return colors


def save_frame(fig, directory: Path, idx: int):
    directory.mkdir(exist_ok=True)
    fig.savefig(directory / f"frame_{idx:03d}.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Dominance illustration
# ---------------------------------------------------------------------------

def save_dominance_illustration():
    fig, ax = plt.subplots(figsize=(6, 5))
    ref = np.array([0.55, 0.55])
    ax.scatter(*ref, s=160, color="#1f77b4", zorder=5, label="точка $x$")

    regions = [
        ((-0.02, 0.55), (0.55, 1.02), "#d62728", 0.25, "доминирует $x$"),
        ((0.55, 1.02), (-0.02, 0.55), "#2ca02c", 0.25, "доминируется $x$"),
        ((0.55, 1.02), (0.55, 1.02), "#aec7e8", 0.18, "несравнимо"),
        ((-0.02, 0.55), (-0.02, 0.55), "#aec7e8", 0.18, None),
    ]
    for (x0, x1), (y0, y1), color, alpha, label in regions:
        ax.fill([x0, x1, x1, x0], [y0, y0, y1, y1], color=color, alpha=alpha,
                label=label, zorder=1)

    ax.axvline(ref[0], color="gray", lw=0.8, ls="--")
    ax.axhline(ref[1], color="gray", lw=0.8, ls="--")
    ax.set_xlabel("$f_1$")
    ax.set_ylabel("$f_2$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Области доминирования")

    handles, labels = ax.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    ax.legend(h2, l2, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "dominance_illustration.png", bbox_inches="tight")
    plt.close(fig)
    print("  dominance_illustration.png")


# ---------------------------------------------------------------------------
# 2. Crowding distance illustration
# ---------------------------------------------------------------------------

def save_crowding_illustration():
    rng = np.random.default_rng(7)
    f1 = np.sort(rng.uniform(0.05, 0.95, 7))
    f2 = 1.0 - f1 + rng.normal(0, 0.02, 7)
    f2 = np.clip(f2, 0.05, 0.95)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(f1, f2, "o-", color="#1f77b4", ms=9, lw=1.2, label="фронт $\\mathcal{F}_i$")

    mid = len(f1) // 2
    ax.annotate("", xy=(f1[mid + 1], f2[mid]), xytext=(f1[mid - 1], f2[mid]),
                arrowprops=dict(arrowstyle="<->", color="#d62728", lw=2))
    ax.annotate("", xy=(f1[mid], f2[mid + 1]), xytext=(f1[mid], f2[mid - 1]),
                arrowprops=dict(arrowstyle="<->", color="#2ca02c", lw=2))

    rect_x = [f1[mid - 1], f1[mid + 1], f1[mid + 1], f1[mid - 1], f1[mid - 1]]
    rect_y = [f2[mid - 1], f2[mid - 1], f2[mid + 1], f2[mid + 1], f2[mid - 1]]
    ax.plot(rect_x, rect_y, "k--", lw=1, alpha=0.6)
    ax.scatter([f1[0], f1[-1]], [f2[0], f2[-1]], s=120, color="#ff7f0e",
               zorder=6, label="граничные точки ($d=\\infty$)")
    ax.scatter(f1[mid], f2[mid], s=120, color="#d62728", zorder=6,
               label="оцениваемая точка $i$")
    ax.text(f1[mid] + 0.01, f2[mid] + 0.04, "$i_{\\mathrm{distance}}$", color="#d62728")

    ax.set_xlabel("$f_1$")
    ax.set_ylabel("$f_2$")
    ax.set_title("Расстояние толпы")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "crowding_illustration.png", bbox_inches="tight")
    plt.close(fig)
    print("  crowding_illustration.png")


# ---------------------------------------------------------------------------
# 3. True Pareto fronts
# ---------------------------------------------------------------------------

def save_true_fronts():
    specs = [
        ("ZDT1", true_front_zdt1(), "zdt1_true_front.png"),
        ("ZDT2", true_front_zdt2(), "zdt2_true_front.png"),
        ("ZDT3", true_front_zdt3(), "zdt3_true_front.png"),
        ("SCH", true_front_sch(), "sch_true_front.png"),
    ]
    for name, tf, fname in specs:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(tf[:, 0], tf[:, 1], lw=2, color="#1f77b4")
        ax.set_xlabel("$f_1$")
        ax.set_ylabel("$f_2$")
        ax.set_title(f"Истинный фронт Парето: {name}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(ASSETS / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname}")


# ---------------------------------------------------------------------------
# 4. Animation frames
# ---------------------------------------------------------------------------

def _evolution_frames(
    history: dict,
    true_front: Array,
    frame_dir: Path,
    title: str,
    n_frames: int = 50,
    every: int | None = None,
):
    n_gen = len(history["objectives"]) - 1
    if every is None:
        every = max(1, n_gen // n_frames)
    gens = list(range(0, n_gen + 1, every))[:n_frames]

    for frame_idx, gen in enumerate(gens):
        F = history["objectives"][gen]
        fronts = history["fronts"][gen]
        colors = _rank_colors(F, fronts)
        front0 = np.array(fronts[0])

        fig, ax = plt.subplots(figsize=(6, 4.8))
        ax.plot(true_front[:, 0], true_front[:, 1], lw=2,
                color="#aaaaaa", label="истинный фронт", zorder=1)
        sc = ax.scatter(F[:, 0], F[:, 1], c=colors, cmap=CMAP,
                        s=22, alpha=0.75, zorder=3, vmin=0, vmax=max(4, colors.max()))
        ax.scatter(F[front0, 0], F[front0, 1], s=40, edgecolors="#1f77b4",
                   facecolors="none", lw=1.4, zorder=4, label="ранг 1")
        ax.set_xlabel("$f_1$")
        ax.set_ylabel("$f_2$")
        ax.set_title(f"{title}  (поколение {gen})")
        ax.legend(fontsize=8, loc="upper right")
        plt.colorbar(sc, ax=ax, label="ранг", shrink=0.8)
        fig.tight_layout()
        save_frame(fig, frame_dir, frame_idx)

    print(f"  {frame_dir.name}: {len(gens)} frames")


# ---------------------------------------------------------------------------
# 5. Convergence plots
# ---------------------------------------------------------------------------

def save_convergence_plots(histories: dict, true_fronts: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    colors = {"ZDT1": "#1f77b4", "ZDT2": "#ff7f0e", "ZDT3": "#2ca02c", "SCH": "#d62728"}

    for name, hist in histories.items():
        tf = true_fronts[name]
        conv = []
        div = []
        for gen in range(len(hist["objectives"])):
            ff = get_first_front(hist, gen)
            if len(ff) >= 2:
                conv.append(convergence_metric(ff, tf))
                div.append(diversity_metric(ff))
            else:
                conv.append(np.nan)
                div.append(np.nan)
        gens = np.arange(len(conv))
        axes[0].plot(gens, conv, label=name, color=colors[name])
        axes[1].plot(gens, div, label=name, color=colors[name])

    axes[0].set_xlabel("поколение")
    axes[0].set_ylabel("$\\Upsilon$ (сходимость)")
    axes[0].set_title("Метрика сходимости")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("поколение")
    axes[1].set_ylabel("$\\Delta$ (разнообразие)")
    axes[1].set_title("Метрика разнообразия")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(ASSETS / "convergence_diversity.png", bbox_inches="tight")
    plt.close(fig)
    print("  convergence_diversity.png")


# ---------------------------------------------------------------------------
# 6. Final fronts comparison
# ---------------------------------------------------------------------------

def save_final_fronts(histories: dict, true_fronts: dict):
    specs = [
        ("ZDT1", "ZDT1 — выпуклый фронт"),
        ("ZDT2", "ZDT2 — невыпуклый фронт"),
        ("ZDT3", "ZDT3 — разрывный фронт"),
        ("SCH", "SCH — 1 переменная"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (name, title) in zip(axes.flat, specs):
        hist = histories[name]
        tf = true_fronts[name]
        ff = get_first_front(hist, -1)
        ax.plot(tf[:, 0], tf[:, 1], lw=2, color="#aaaaaa", label="истинный фронт")
        ax.scatter(ff[:, 0], ff[:, 1], s=18, color="#1f77b4", label="NSGA-II", zorder=3)
        ax.set_xlabel("$f_1$")
        ax.set_ylabel("$f_2$")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "final_fronts.png", bbox_inches="tight")
    plt.close(fig)
    print("  final_fronts.png")


# ---------------------------------------------------------------------------
# 7. NSGA-II procedure diagram
# ---------------------------------------------------------------------------

def save_procedure_diagram():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, color, label, fontsize=10):
        rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                       boxstyle="round,pad=0.1",
                                       facecolor=color, edgecolor="black", lw=1.2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, wrap=True)

    def arrow(x1, x2, y):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    box(1.0, 2.0, 1.6, 1.0, "#aec7e8", "$P_t$")
    box(1.0, 0.8, 1.6, 0.8, "#ffbb78", "$Q_t$")

    ax.annotate("", xy=(2.3, 2.0), xytext=(1.8, 2.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.annotate("", xy=(2.3, 1.4), xytext=(1.8, 1.4),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(2.5, 1.7, "$R_t = P_t \\cup Q_t$\n$(2N)$",
            ha="center", va="center", fontsize=9)

    box(4.0, 1.7, 1.8, 1.2, "#c5b0d5",
        "Недоминируемая\nсортировка\n$\\mathcal{F}_1, \\mathcal{F}_2, \\ldots$", fontsize=9)
    arrow(3.5, 3.1, 1.7)

    box(6.3, 2.5, 1.6, 0.9, "#98df8a", "$\\mathcal{F}_1$", 10)
    box(6.3, 1.6, 1.6, 0.7, "#98df8a", "$\\mathcal{F}_2$", 10)
    box(6.3, 0.9, 1.6, 0.6, "#d9d9d9", "$\\mathcal{F}_l$ (краудинг)", 8)
    for yf in [2.5, 1.6, 0.9]:
        arrow(4.9, 5.5, yf)

    box(8.2, 1.7, 1.2, 1.8, "#aec7e8", "$P_{t+1}$\n$(N)$", 10)
    for yf in [2.5, 1.6]:
        arrow(7.1, 7.6, yf)
    ax.annotate("", xy=(7.6, 0.9), xytext=(7.1, 0.9),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5, linestyle="dashed"))

    ax.text(4.5, 3.7, "Шаг $t \\to t+1$", ha="center", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "nsga2_procedure.png", bbox_inches="tight")
    plt.close(fig)
    print("  nsga2_procedure.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("Generating static illustrations...")
    save_dominance_illustration()
    save_crowding_illustration()
    save_procedure_diagram()
    save_true_fronts()

    print("\nRunning NSGA-II on test problems...")
    cfg_main = NSGAConfig(pop_size=100, n_gen=250, seed=42)
    cfg_sch = NSGAConfig(pop_size=50, n_gen=100, seed=1)

    problems = {
        "ZDT1": zdt1(30),
        "ZDT2": zdt2(30),
        "ZDT3": zdt3(30),
        "SCH": sch(),
    }
    true_fronts = {
        "ZDT1": true_front_zdt1(500),
        "ZDT2": true_front_zdt2(500),
        "ZDT3": true_front_zdt3(500),
        "SCH": true_front_sch(200),
    }
    histories = {}
    for name, (objs, bounds) in problems.items():
        cfg = cfg_sch if name == "SCH" else cfg_main
        print(f"  {name}...", end=" ", flush=True)
        histories[name] = nsga2(objs, bounds, cfg)
        print("done")

    print("\nGenerating animation frames...")
    frame_specs = [
        ("ZDT1", "frames_zdt1", true_fronts["ZDT1"], "ZDT1"),
        ("ZDT2", "frames_zdt2", true_fronts["ZDT2"], "ZDT2"),
        ("ZDT3", "frames_zdt3", true_fronts["ZDT3"], "ZDT3"),
        ("SCH",  "frames_sch",  true_fronts["SCH"],  "SCH"),
    ]
    for name, dirname, tf, label in frame_specs:
        _evolution_frames(histories[name], tf, ASSETS / dirname, label, n_frames=25)

    print("\nGenerating comparison plots...")
    save_convergence_plots(histories, true_fronts)
    save_final_fronts(histories, true_fronts)

    print("\nDone. All assets saved to", ASSETS)


if __name__ == "__main__":
    main()
