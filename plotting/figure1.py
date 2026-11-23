"""Figure 1, 210 x 68 mm.

a, b (problem): the real agent loop. A planner fixes a finite set of states; the module studied here
      chooses the next state (LLM answer -> parser s_i -> state j); an executor carries it out and the
      agent returns to the module from the new state, so repeated choices form a Markov chain.
      Dark node = current state; coloured = reachable in one step (dashed arrows); grey = not reachable.
c (mechanism, schematic): a five-state ring whose reference kernel Q circulates (pure loop, no
      potential). Post-training reweights it, T = Q exp(u/beta)/Z. Node height = potential part of
      y_ij = log(T_ij/T_ji) (common scale); arrows = net flow y_ij (width ~ |y_ij|).
      Bottom: r(beta) of the same toy model; three equally spaced guides mark the snapshots.
Panels a,b layout chosen by side-by-side comparison of variants A (filled boxes, straight arrows), B (no boxes) and C (outlined ends, grey module card, arc cycle; default).
"""
import sys, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "analysis"))
from style import setup, BLUE, ORANGE, INK, MUTED, GRID, MM
from common import task_spec, hodge

OUT = Path(os.environ.get("PFL_FIGURES", HERE.parent / "figures")); OUT.mkdir(parents=True, exist_ok=True)
VAR = os.environ.get("FIG1_AB", "C")
setup()
W, H = 210, 68           # layout units; x is drawn at 183 mm (the width of Figs 2-5) so fonts print at true size
FW = 183
KX = FW / W              # mm per x layout unit
fig = plt.figure(figsize=(FW * MM, H * MM))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")
cmap1 = LinearSegmentedColormap.from_list("b", ["#E3EEF6", "#7FB3D8", BLUE, "#00385A"])
cmap2 = LinearSegmentedColormap.from_list("o", ["#FBEAE0", "#F0A27A", ORANGE, "#6E2A00"])
SOFT = "#F4F6F8"


def arr(p0, p1, rad=0.0, color=MUTED, lw=0.7, ms=6.5):
    bg.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                 mutation_scale=ms, lw=lw, color=color, shrinkA=0, shrinkB=0))


def lab(x, y, s, **kw):
    kw.setdefault("fontsize", 5.8); kw.setdefault("color", MUTED); kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    bg.text(x, y, s, **kw)


def graph(task, cx, cy, R, cmap, color):
    u, _, _ = task_spec(task); n = len(u)
    th = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    pos = np.c_[cx + R / KX * np.cos(th), cy + R * np.sin(th)]
    if task == "T1":
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]; cand = list(range(1, n))
    else:
        edges = [(i, (i + d) % n) for i in range(n) for d in (1, 2)]; cand = [d % n for d in (-2, -1, 1, 2)]
    for i, j in edges:
        bg.plot(*np.c_[pos[i], pos[j]], color=GRID, lw=0.4, zorder=1)
    for j in cand:
        bg.add_patch(FancyArrowPatch(pos[0], pos[j], arrowstyle="-|>", mutation_scale=4.5, lw=0.55, color=color,
                                     ls=(0, (2, 1.2)), shrinkA=2.8, shrinkB=2.6, zorder=2))
    lo, hi = u.min(), u.max()
    for i, p in enumerate(pos):
        cur, act = i == 0, i == 0 or i in cand
        fc = INK if cur else color if act else "#EEF1F4"
        bg.scatter(*p, s=26 if cur else 15 if act else 9, c=[fc], edgecolors="white" if act else "#B4BEC6",
                   linewidths=0.7 if act else 0.4, zorder=3)
    return n, len(cand)


rows = [("a", "Safety-testing agent (T1)", "T1",
         ("Planner", "8 test cases\n(carrier × pretext\n× probe)"), "choose next test",
         ("Executor", "runs the test\non the office\nassistant"), cmap1, BLUE, 49.0),
        ("b", "Physics-solving agent (T2)", "T2",
         ("Planner", "12 checkpoints of\na two-layer heat-\nconduction problem"), "choose next step",
         ("Executor", "carries out\nthe chosen step"), cmap2, ORANGE, 15.5)]

for letter, name, task, plan, title, ex, cmap, color, yc in rows:
    bg.text(2, yc + 16.0, letter, fontsize=8, fontweight="bold", va="center")
    bg.text(6.5, yc + 16.0, name, fontsize=7, fontweight="bold", va="center")
    if VAR == "A":          # boxed text ends, straight two-arrow cycle between module and executor
        def tbox(x, w, head, body):
            bg.add_patch(FancyBboxPatch((x, yc - 8), w, 16, boxstyle="round,pad=0,rounding_size=1.4",
                                        fc=SOFT, ec="none"))
            bg.text(x + w / 2, yc + 4.6, head, ha="center", va="center", fontsize=6.3, fontweight="bold")
            bg.text(x + w / 2, yc - 1.4, body, ha="center", va="center", fontsize=5.6, linespacing=1.35)
        tbox(3, 28, *plan)
        arr((32, yc), (40, yc)); lab(36, yc + 2.0, "states")
        cx = 55.5
        bg.add_patch(FancyBboxPatch((cx - 14.5, yc - 12), 29, 24.5, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc="white", ec=color, lw=0.9))
        bg.text(cx, yc + 9.7, title, ha="center", va="center", fontsize=6.3, fontweight="bold")
        graph(task, cx, yc + 0.6, 6.0, cmap, color)
        lab(cx, yc - 9.3, "answer → $s_i$ → state $j$", fontsize=5.5, color=INK)
        arr((71, yc + 3.2), (79, yc + 3.2)); lab(75, yc + 5.4, "go")
        arr((79, yc - 3.2), (71, yc - 3.2)); lab(75, yc - 5.4, "new state")
        tbox(80, 26, *ex)
    elif VAR == "B":        # no boxes on the ends, curved cycle under the module
        def tplain(x, head, body):
            bg.text(x, yc + 3.6, head, ha="center", va="center", fontsize=6.3, fontweight="bold")
            bg.text(x, yc - 1.8, body, ha="center", va="center", fontsize=5.6, linespacing=1.35)
        tplain(17, *plan)
        arr((33, yc + 0.5), (40, yc + 0.5))
        cx = 55.5
        bg.add_patch(FancyBboxPatch((cx - 14.5, yc - 12), 29, 24.5, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc="white", ec=color, lw=0.9))
        bg.text(cx, yc + 9.7, title, ha="center", va="center", fontsize=6.3, fontweight="bold")
        graph(task, cx, yc + 0.6, 6.0, cmap, color)
        lab(cx, yc - 9.3, "answer → $s_i$ → state $j$", fontsize=5.5, color=INK)
        arr((71, yc + 0.5), (79, yc + 0.5))
        tplain(93, *ex)
        arr((93, yc - 6.5), (cx + 4, yc - 12.6), rad=-0.35); lab(84, yc - 13.2, "repeat")
    else:                   # C: module centred, planner above-left, executor in a closed loop
        # printed sizes: headings 7 pt, body 6.5 pt, as in Figs 2-5
        def tbox(x, w, head, body):
            bg.add_patch(FancyBboxPatch((x, yc - 10), w, 20, boxstyle="round,pad=0,rounding_size=1.4",
                                        fc="white", ec=GRID, lw=0.6))
            bg.text(x + w / 2, yc + 6.4, head, ha="center", va="center", fontsize=7, fontweight="bold")
            bg.text(x + w / 2, yc - 1.6, body, ha="center", va="center", fontsize=6.5, linespacing=1.3)
        tbox(2, 30, *plan)
        arr((33, yc), (40, yc))
        cx = 56.5
        bg.add_patch(FancyBboxPatch((cx - 15.5, yc - 12.5), 31, 25.5, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc=SOFT, ec="none"))
        bg.text(cx, yc + 10.0, title, ha="center", va="center", fontsize=7, fontweight="bold", color=color)
        graph(task, cx, yc + 0.3, 6.0, cmap, color)
        lab(cx, yc - 9.6, "answer → $s_i$ → state $j$", fontsize=6.5, color=INK)
        arr((73.5, yc + 3.5), (81.0, yc + 3.5), rad=-0.45)
        arr((81.0, yc - 3.5), (73.5, yc - 3.5), rad=-0.45)
        tbox(82.5, 25, *ex)
# ---------------- c: mechanism (schematic toy model) ----------------
n = 5
uu = np.array([0.0, 0.5, 1.0, 0.55, 0.2])     # ring order = left-to-right order
E = sorted(tuple(sorted((i, (i + 1) % n))) for i in range(n))   # fixed orientation i<j, as for E_*
p_fwd = 0.85                                       # Q circulates i -> i+1: a pure loop, no potential
Q = np.zeros((n, n))
for i in range(n):
    Q[i, (i + 1) % n], Q[i, (i - 1) % n] = p_fwd, 1 - p_fwd


def parts(beta):
    Wt = np.where(Q > 0, Q * np.exp((uu - uu.max()) / beta), 0); T = Wt / Wt.sum(1, keepdims=True)
    y = np.array([np.log(T[i, j] / T[j, i]) for i, j in E])
    phi, g, res = hodge(y, E, n)
    r = np.corrcoef(g, y)[0, 1] if np.ptp(g) > 1e-9 else 0.0
    return y, phi, r


snaps = np.array([-0.8, 0.7, 2.2])                 # equally spaced in -log(beta)
X0, X1, XL, XR = 124.0, 206.0, -2.0, 3.0           # r-axis extent (mm) and limits
mm = lambda lb: X0 + (lb - XL) / (XR - XL) * (X1 - X0)
centres = [mm(s) for s in snaps]
lbs = np.linspace(XL, XR, 100)
rs = np.array([parts(np.exp(-lb))[2] for lb in lbs])
heights = [parts(np.exp(-lb))[1] for lb in snaps]
hs = 11.0 / max(np.ptp(h) for h in heights)
slot = {i: i for i in range(n)}

YB, YT = 6.0, 27.0
axr = fig.add_axes([X0 / W, YB / H, (X1 - X0) / W, (YT - YB) / H]); axr.set_facecolor("none")
axr.fill_between(lbs, rs, -0.12, color=BLUE, alpha=0.07, lw=0)
axr.axhline(1, color=GRID, lw=0.6, ls=(0, (3, 2)), zorder=0)
axr.plot(lbs, rs, color=BLUE, lw=1.5, solid_capstyle="round")
axr.set_xlim(XL, XR); axr.set_ylim(-0.12, 1.1); axr.set_yticks([0, 1]); axr.set_xticks([])
axr.set_ylabel("order parameter $r$", labelpad=1)
for sp in ("top", "right"):
    axr.spines[sp].set_visible(True)
for sp in axr.spines.values():
    sp.set_color("#AEB9C2"); sp.set_linewidth(0.6)
axr.tick_params(length=0, pad=2)
axr.set_xlabel("post-training  ($-\\log\\beta$)", labelpad=3)
CB, CH, CW = 35.5, 25.0, 22.8                      # card bottom, height, width (mm)
grad = LinearSegmentedColormap.from_list("g", ["#D6E6F2", "#F7FAFC"])
for ks, (lb, cxs, phi) in enumerate(zip(snaps, centres, heights)):
    y, _, r = parts(np.exp(-lb))
    axr.scatter(lb, r, s=24, c=[BLUE], edgecolors="white", linewidths=1.0, zorder=4)
    rmm = YB + (r + 0.12) / 1.22 * (YT - YB)
    bg.plot([cxs, cxs], [rmm + 1.4, CB], color=MUTED, lw=0.5, ls=(0, (1, 1.6)), zorder=0)
    card = FancyBboxPatch((cxs - CW / 2, CB), CW, CH, boxstyle="round,pad=0,rounding_size=1.6",
                          fc="#FBFCFD", ec="#C9D3DB", lw=0.6, zorder=0)
    bg.add_patch(card)
    hh = (phi - phi.mean()) * hs
    yc0 = CB + CH / 2 - (hh.max() + hh.min()) / 2 + 1.0
    pos = np.array([[cxs - 9.0 + 4.5 * slot[i], yc0 + hh[i]] for i in range(n)])
    xf = np.linspace(cxs - CW / 2, cxs + CW / 2, 200)
    f = PchipInterpolator(pos[:, 0], pos[:, 1], extrapolate=True)
    yl = np.minimum(f(xf) - 1.1, CB + CH)
    poly = np.r_[np.c_[xf, yl], [[xf[-1], CB], [xf[0], CB]]]
    im = bg.imshow(np.linspace(1, 0, 64)[:, None], extent=(xf[0], xf[-1], CB, CB + CH), cmap=grad,
                   aspect="auto", zorder=1, interpolation="bicubic")
    clip = plt.Polygon(poly, closed=True, transform=bg.transData)
    im.set_clip_path(card)
    top = np.r_[np.c_[xf, yl], [[xf[-1], CB + CH], [xf[0], CB + CH]]]
    cover = plt.Polygon(top, closed=True, fc="#FBFCFD", ec="none", zorder=1.5)
    bg.add_patch(cover); cover.set_clip_path(card)
    bg.add_patch(FancyBboxPatch((cxs - CW / 2, CB), CW, CH, boxstyle="round,pad=0,rounding_size=1.6",
                                fc="none", ec="#C9D3DB", lw=0.6, zorder=6))
    bg.set_xlim(0, W); bg.set_ylim(0, H)
    ln, = bg.plot(xf, yl, color=BLUE, lw=1.0, alpha=0.75, zorder=2)
    ln.set_clip_path(card)
    for k_, (i, j) in enumerate(E):
        a, b = (i, j) if y[k_] > 0 else (j, i)
        if abs(i - j) > 1:
            # closing edge of the loop: a solid return arc above the nodes while the loop still carries
            # the flow; omitted at the last (r ~ 1) snapshot, where the potential controls
            if ks < len(snaps) - 1:
                bg.add_patch(FancyArrowPatch(pos[a] + [0, 1.4], pos[b] + [0, 1.4], connectionstyle="arc3,rad=0.42", arrowstyle="-|>",
                                             mutation_scale=5.0, lw=0.5 + 0.6 * min(1, abs(y[k_]) / 6),
                                             color="#46576A", shrinkA=1.0, shrinkB=1.0, zorder=4))
            continue
        if abs(y[k_]) < 0.3:
            continue
        rad = -0.45 if pos[b, 0] > pos[a, 0] else 0.45
        bg.add_patch(FancyArrowPatch(pos[a], pos[b], connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                     mutation_scale=5.0, lw=0.5 + 0.6 * min(1, abs(y[k_]) / 6), color="#46576A",
                                     shrinkA=2.2, shrinkB=2.2, zorder=4, capstyle="round"))
    for i in range(n):
        bg.scatter(*pos[i], s=22, c=[BLUE], edgecolors="white", linewidths=0.8, zorder=5)
bg.text(122, 65.5, "c", fontsize=8, fontweight="bold", va="center")
bg.text(126.5, 65.5, "Post-training makes a potential emerge", fontsize=7, fontweight="bold", va="center")

suffix = "" if "FIG1_AB" not in os.environ else f"_{VAR}"
fig.savefig(OUT / f"theory_experiment_overview{suffix}.pdf")
fig.savefig(OUT / f"theory_experiment_overview{suffix}.png", dpi=220)
print("r at snaps:", [round(parts(np.exp(-lb))[2], 2) for lb in snaps])
