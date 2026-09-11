import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from functools import partial
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss
from pbh17 import L_Edd, v_L_rms, mass_labels, setup
import single_branch_luminosity as single_branch
from radiation_pressure_single_branch import (
    z_vals, fig_masses, L_collisional, L_photoionization,
)

def feedback_L(M, z, v_rel=0, luminosity_func=single_branch.L):
    Ledd = L_Edd(M)
    L0 = float(luminosity_func(M, z, v_rel))
    if not np.isfinite(L0) or L0 < 0:
        raise ValueError("The luminosity must be finite and nonnegative.")
    if L0 == 0:
        return 0.0

    eddington_bound = (1 - 1e-12) * Ledd / L0
    upper = min(1.0, eddington_bound)
    
    def residual(f):
        M_eff = M * (1 - f * L0 / Ledd)
        return f - float(luminosity_func(M_eff, z, v_rel)) / L0
    error = residual(upper)
    if abs(error) < 1e-9:
        return L0 * upper
    if error < 0:
        upper = eddington_bound
    return L0 * brentq(residual, 0.0, upper, xtol=1e-14, rtol=1e-9)

def average_feedback_L(M, z, luminosity_func=single_branch.L, nv=64, radiation_pressure=True):
    if nv < 2:
        raise ValueError("nv must be at least 2.")
    L = partial(feedback_L, luminosity_func=luminosity_func) if radiation_pressure else luminosity_func
    sigma = float(v_L_rms(z)) / np.sqrt(3)
    if sigma == 0:
        return float(L(M, z, 0))
    nodes, weights = leggauss(nv)
    
    q = 4 * (nodes + 1)
    weights = 4 * weights * np.sqrt(2 / np.pi) * q**2 * np.exp(-q**2 / 2)
    weights /= weights.sum()
    luminosities = [L(M, z, sigma * qi) for qi in q]
    return float(np.dot(weights, luminosities))

def _average_luminosity_task(args):
    return average_feedback_L(*args)

def solve_average_redshift_curve(M, z_grid=None, luminosity_func=single_branch.L, nv=64, workers=1, radiation_pressure=True):
    z = np.asarray(z_vals if z_grid is None else z_grid)
    tasks = [(M, zi, luminosity_func, nv, radiation_pressure) for zi in z]
    if workers == 1:
        mean_L = np.array(list(map(_average_luminosity_task, tasks)))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            mean_L = np.array(list(pool.map(_average_luminosity_task, tasks)))
    return {"z": z, "average_L": mean_L, "average_L_over_Ledd": mean_L / L_Edd(M)}

def plot_single_branch_average_luminosity_vs_z(masses=fig_masses, z_grid=None, nv=64, workers=None):
    fig = plt.figure(figsize=(7.5, 5.2))
    for M in masses:
        sol = solve_average_redshift_curve(M, z_grid, nv=nv, workers=workers)
        plt.loglog(sol["z"], sol["average_L_over_Ledd"], linewidth=2, label=mass_labels[M])
    setup("Velocity-averaged luminosity with radiation pressure: single-branch",
          r"$\langle L\rangle/L_{\rm Edd}$", (1e2, 1e5), (1e-16, 2))
    return fig

def plot_average_luminosity_without_feedback(masses=fig_masses, z_grid=None, nv=64, workers=None):
    z_grid = np.logspace(np.log10(30), 5, 160) if z_grid is None else np.asarray(z_grid)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    treatments = ((L_collisional, ":", "collisional"),
                  (single_branch.L, "-", "single-branch"),
                  (L_photoionization, "--", "photoionization"))
    mass_lines = []
    for i, M in enumerate(masses):
        color = colors[i % len(colors)]
        for func, style, label in treatments:
            sol = solve_average_redshift_curve(M, z_grid, func, nv, workers, radiation_pressure=False)
            line, = ax.loglog(sol["z"], sol["average_L_over_Ledd"], style, color=color, linewidth=2)
            if style == "-":
                mass_lines.append(line)
    mass_legend = ax.legend(mass_lines, [mass_labels[M] for M in masses], loc="lower left")
    ax.add_artist(mass_legend)
    ax.legend([Line2D([], [], color="black", linestyle=style, linewidth=2) for _, style, _ in treatments],
              [label for _, _, label in treatments], loc="upper right")
    ax.set_xlabel(r"$z$")
    ax.set_xlim(z_grid[0], z_grid[-1])
    ax.set_ylabel(r"$\langle L\rangle/L_{\rm Edd}$")
    ax.set_title("Velocity-averaged luminosity without radiation pressure")
    fig.tight_layout()
    return fig

def main():
    plot_single_branch_average_luminosity_vs_z()
    plot_average_luminosity_without_feedback()
    plt.show()

if __name__ == "__main__":
    main()
