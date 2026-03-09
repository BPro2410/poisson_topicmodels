"""
Render the directed graphical model (plate diagram) for the CSPF2 model.

Covariate Seeded Poisson Factorization with grouped design-adaptive shrinkage.
Produces: data/figures/CSPF2_graphical_model.pdf  (and .png)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

# ── figure setup ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(16, 14))
ax.set_xlim(-6, 12)
ax.set_ylim(-1.5, 12)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

# ── drawing helpers ───────────────────────────────────────────────────────────
R = 0.42  # default node radius


def draw_node(x, y, label, style="latent", r=R):
    """Draw a single node.
    style: 'latent' (open), 'obs' (shaded), 'det' (double circle), 'param' (dot)
    """
    if style == "param":
        ax.plot(x, y, "ko", markersize=5, zorder=5)
        return
    fc = "white" if style in ("latent", "det") else "#c8c8c8"
    circ = plt.Circle((x, y), r, fc=fc, ec="black", lw=1.6, zorder=4)
    ax.add_patch(circ)
    if style == "det":
        circ2 = plt.Circle((x, y), r * 0.82, fc=fc, ec="black", lw=1.2, zorder=4)
        ax.add_patch(circ2)
    ax.text(x, y, label, ha="center", va="center", fontsize=9, zorder=5,
            math_fontfamily="cm")


def arrow(x1, y1, x2, y2, r_from=R, r_to=R, shrink_extra=0.02):
    """Draw an arrow between two nodes, clipping at circle edges."""
    dx, dy = x2 - x1, y2 - y1
    dist = np.hypot(dx, dy)
    if dist < 1e-6:
        return
    ux, uy = dx / dist, dy / dist
    sx = x1 + ux * (r_from + shrink_extra)
    sy = y1 + uy * (r_from + shrink_extra)
    ex = x2 - ux * (r_to + shrink_extra)
    ey = y2 - uy * (r_to + shrink_extra)
    ax.annotate(
        "",
        xy=(ex, ey),
        xytext=(sx, sy),
        arrowprops=dict(arrowstyle="->,head_width=0.18,head_length=0.14",
                        lw=1.3, color="black"),
        zorder=3,
    )


def draw_plate(x, y, w, h, label, color, label_pos="se"):
    """Draw a dashed plate rectangle with label."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        linewidth=1.5,
        edgecolor=color,
        facecolor=(*plt.matplotlib.colors.to_rgb(color), 0.06),
        linestyle="--",
        zorder=1,
    )
    ax.add_patch(rect)
    positions = {
        "se": (x + w - 0.08, y + 0.15),
        "sw": (x + 0.08, y + 0.15),
        "nw": (x + 0.08, y + h - 0.15),
        "ne": (x + w - 0.08, y + h - 0.15),
    }
    tx, ty = positions[label_pos]
    ha = "right" if "e" in label_pos else "left"
    ax.text(tx, ty, label, fontsize=10, color=color, ha=ha, va="bottom",
            fontstyle="italic", zorder=2)


def param_label(x, y, label, offset=(0, 0.35), fontsize=8, ha="center"):
    """Draw a fixed-parameter dot with a label offset."""
    draw_node(x, y, "", style="param")
    ax.text(x + offset[0], y + offset[1], label, ha=ha, va="center",
            fontsize=fontsize, zorder=5, math_fontfamily="cm")


# ── coordinates ───────────────────────────────────────────────────────────────
# Layer 1: Shrinkage hyperparameters
rhotau_pos   = (0, 9.2)
tau_pos      = (0, 7.7)
rhodelta_pos = (4.5, 9.2)
delta_pos    = (4.5, 7.7)

# Layer 2: Regression coefficients
lambda0_pos  = (-3, 5.2)
lambdagk_pos = (2.5, 6.0)
Xg_pos       = (5.8, 6.0)

# Layer 3: Document–topic link
xd_pos    = (-3, 3.2)
mu_pos    = (0.3, 3.2)
theta_pos = (0.3, 1.6)

# Layer 4: Topic–word (seed decomposition)
bstar_pos  = (7, 3.2)
btilde_pos = (9.5, 3.2)
beta_pos   = (8.25, 1.6)

# Layer 5: Observation
y_pos = (4, 0)


# ── plates (draw first, behind everything) ────────────────────────────────────
#        x      y      w      h
# k plate: topics — contains τ, δ, λ, μ, θ, β but NOT y_{dv}
draw_plate(-4.8, 0.8, 16.0, 10.4,
           r"$k = 1, \ldots, K$", "darkorange", "nw")     # Topic plate
# g plate: groups — contains ρ^(δ), δ², λ_{gk}, X_g
draw_plate(1.7, 5.2, 5.8, 5.8,
           r"$g = 1, \ldots, G$", "royalblue", "se")       # Group plate
# d plate: documents — contains x_d, μ, θ, y_{dv}
draw_plate(-3.8, -0.7, 9.3, 4.8,
           r"$d = 1, \ldots, D$", "forestgreen", "sw")     # Document plate
# v plate: vocabulary — contains β*, β̃, β, y_{dv}
draw_plate(3.2, -0.7, 7.3, 6.0,
           r"$v = 1, \ldots, V$", "firebrick", "se")       # Vocabulary plate


# ── nodes ─────────────────────────────────────────────────────────────────────

# --- Global shrinkage ---
param_label(-0.65, 10.3, r"$a_{\rho\tau}$", offset=(0, 0.32))
param_label( 0.65, 10.3, r"$b_{\rho\tau}$", offset=(0, 0.32))
draw_node(*rhotau_pos, r"$\rho_k^{(\tau)}$", "latent")
param_label(-1.4, 7.7, r"$a_{\tau}$", offset=(-0.35, 0), ha="right")
draw_node(*tau_pos, r"$\tau_k^2$", "latent")

# --- Local shrinkage ---
param_label(3.85, 10.3, r"$a_{\rho\delta}$", offset=(0, 0.32))
param_label(5.15, 10.3, r"$b_{\rho\delta}$", offset=(0, 0.32))
draw_node(*rhodelta_pos, r"$\rho_{gk}^{(\delta)}$", "latent")
param_label(6.1, 7.7, r"$a_{\delta}$", offset=(0.35, 0), ha="left")
draw_node(*delta_pos, r"$\delta_{gk}^2$", "latent")

# --- Intercept ---
param_label(-3, 6.3, r"$s^2_{\lambda_0}$", offset=(0, 0.32))
draw_node(*lambda0_pos, r"$\lambda_{0k}$", "latent")

# --- Group coefficients ---
draw_node(*lambdagk_pos, r"$\lambda_{gk}$", "latent")
draw_node(*Xg_pos, r"$\mathbf{X}_g$", "obs")

# --- Document-topic link ---
draw_node(*xd_pos, r"$\mathbf{x}_d$", "obs")
draw_node(*mu_pos, r"$\mu_{\theta,kd}$", "det")
param_label(-1.3, 1.6, r"$b_{\theta}$", offset=(-0.35, 0), ha="right")
draw_node(*theta_pos, r"$\theta_{dk}$", "latent")

# --- Topic-word priors ---
param_label(7, 4.5, r"$a_{\beta^\star}, b_{\beta^\star}$", offset=(0, 0.32))
draw_node(*bstar_pos, r"$\beta_{kv}^\star$", "latent")
param_label(9.5, 4.5, r"$a_{\tilde{\beta}}, b_{\tilde{\beta}}$", offset=(0, 0.32))
draw_node(*btilde_pos, r"$\tilde{\beta}_{kv}$", "latent")
draw_node(*beta_pos, r"$\beta_{kv}$", "det")

# --- Observation ---
draw_node(*y_pos, r"$y_{dv}$", "obs")


# ── edges ─────────────────────────────────────────────────────────────────────
PR = 0.04  # param radius (dots)

# Global shrinkage chain
arrow(-0.65, 10.3, *rhotau_pos, r_from=PR)
arrow( 0.65, 10.3, *rhotau_pos, r_from=PR)
arrow(*rhotau_pos, *tau_pos)
arrow(-1.4, 7.7, *tau_pos, r_from=PR)

# Local shrinkage chain
arrow(3.85, 10.3, *rhodelta_pos, r_from=PR)
arrow(5.15, 10.3, *rhodelta_pos, r_from=PR)
arrow(*rhodelta_pos, *delta_pos)
arrow(6.1, 7.7, *delta_pos, r_from=PR)

# Shrinkage → group coefficients
arrow(*tau_pos, *lambdagk_pos)
arrow(*delta_pos, *lambdagk_pos)
arrow(*Xg_pos, *lambdagk_pos)

# Intercept prior
arrow(-3, 6.3, *lambda0_pos, r_from=PR)

# Coefficients → deterministic link
arrow(*lambda0_pos, *mu_pos)
arrow(*lambdagk_pos, *mu_pos)
arrow(*xd_pos, *mu_pos)

# Link → document-topic intensity
arrow(*mu_pos, *theta_pos)
arrow(-1.3, 1.6, *theta_pos, r_from=PR)

# Document-topic → observation
arrow(*theta_pos, *y_pos)

# Topic-word priors
arrow(7, 4.5, *bstar_pos, r_from=PR)
arrow(9.5, 4.5, *btilde_pos, r_from=PR)

# Seed decomposition → deterministic sum
arrow(*bstar_pos, *beta_pos)
arrow(*btilde_pos, *beta_pos)

# Topic-word → observation
arrow(*beta_pos, *y_pos)


# ── legend ────────────────────────────────────────────────────────────────────
legend_x, legend_y = -5.5, 11.5
legend_items = [
    ("obs",    "Observed"),
    ("latent", "Latent"),
    ("det",    "Deterministic"),
    ("param",  "Fixed parameter"),
]
for i, (style, text) in enumerate(legend_items):
    ly = legend_y - i * 0.6
    draw_node(legend_x, ly, "", style=style, r=0.18)
    ax.text(legend_x + 0.4, ly, text, va="center", fontsize=8)

# box around legend
legend_box = FancyBboxPatch(
    (legend_x - 0.38, legend_y - 2.1), 2.5, 2.7,
    boxstyle="round,pad=0.08", lw=0.8,
    ec="gray", fc="white", zorder=0,
)
ax.add_patch(legend_box)


# ── save ──────────────────────────────────────────────────────────────────────
out_base = "data/figures/CSPF2_graphical_model"
fig.savefig(f"{out_base}.pdf", bbox_inches="tight", dpi=300)
fig.savefig(f"{out_base}.png", bbox_inches="tight", dpi=300)
print(f"Saved  {out_base}.pdf  and  {out_base}.png")
plt.close(fig)
