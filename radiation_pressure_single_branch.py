'''
Author: Roy Chowdhury
Description: We couple radiation-pressure feedback to the single-branch luminosity model. Radiation pressure reduces the effective gravitational
mass experienced by the infalling gas by a factor of (1 - L/L_Edd), while the luminosity at each effective mass is obtained from the
self-consistent temperature satisfying T_S_func(R_func(T_S)) = T_S. The coupled equation L_single(M_eff(L)) = L is solved iteratively.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from functools import partial
from pbh17 import *
import single_branch_luminosity as single_branch

M_vals = np.logspace(0, 5, 220)
z_vals = np.logspace(2, 5, 120)
fig_masses = [1, 1e2, 1e4, 1e5]
L_collisional = partial(single_branch.L, R=0)
L_photoionization = partial(single_branch.L, R=np.inf)

def clean_L(Lval, Lmin, Lmax):
    if Lval <= 0: 
        return Lmin
    if Lval >= Lmax: 
        return Lmax
    return Lval

def luminosity_error(M, L_guess, z, v_rel=0, luminosity_func=single_branch.L):
    Ledd = L_Edd(M)
    Lmin, Lmax = 1e-30 * Ledd, 0.999 * Ledd
    M_eff = M * (1 - L_guess / Ledd)
    L_new = clean_L(luminosity_func(M_eff, z, v_rel), Lmin, Lmax)
    residual = (L_new - L_guess) / Ledd
    return L_new, residual, M_eff

def luminosity_iteration(M, z, L_initial=None, v_rel=0, luminosity_func=single_branch.L):
    Ledd = L_Edd(M)
    Lmin, Lmax = 1e-30 * Ledd, 0.999 * Ledd
    L_guess = 0.5 * Ledd if L_initial is None else clean_L(L_initial, Lmin, Lmax)
    damping, max_iter = (0.005, 4000) if M >= 1e5 else ((0.01, 2500) if M >= 1e4 else (0.05, 1000))
    for i in range(max_iter):
        L_guess = clean_L(L_guess, Lmin, Lmax)
        L_new, residual, M_eff = luminosity_error(M, L_guess, z, v_rel, luminosity_func)
        if abs(residual) < 0.01: 
            break
        L_guess = (1 - damping) * L_guess + damping * L_new
    return {"L": L_new, "L_over_Ledd": L_new/Ledd, "M_eff": M_eff, "M_eff_over_M": M_eff/M, "residual": residual, "iterations": i+1}

def solve_redshift_curve(M, z_grid=None, v_rel=0, luminosity_func=single_branch.L):
    z_grid = z_vals if z_grid is None else z_grid
    L_initial, L_over_Ledd, M_eff_over_M = None, [], []
    for z in z_grid:
        sol = luminosity_iteration(M, z, L_initial, v_rel, luminosity_func)
        L_initial = sol["L"]
        L_over_Ledd.append(sol["L_over_Ledd"]); M_eff_over_M.append(sol["M_eff_over_M"])
    return {"z": z_grid, "L_over_Ledd": np.array(L_over_Ledd), "M_eff_over_M": np.array(M_eff_over_M)}

def solve_Meff_curve(z, M_grid=None, v_rel=0):
    M_grid = M_vals if M_grid is None else M_grid
    L_initial, M_eff, M_eff_over_M = None, [], []
    for M in M_grid:
        sol = luminosity_iteration(M, z, L_initial, v_rel)
        L_initial = sol["L"]
        M_eff.append(sol["M_eff"]); M_eff_over_M.append(sol["M_eff_over_M"])
    return {"M": M_grid, "M_eff": np.array(M_eff), "M_eff_over_M": np.array(M_eff_over_M)}

def add_branched_legends(masses, colors):
    mass_legend = plt.legend(
        [Line2D([0], [0], color=color, linewidth=2) for color in colors[:len(masses)]],
        [mass_labels[M] for M in masses], loc="lower left"
    )
    plt.gca().add_artist(mass_legend)
    plt.legend(
        [
            Line2D([0], [0], color="black", linestyle=":", linewidth=2),
            Line2D([0], [0], color="black", linestyle="--", linewidth=2),
        ],
        ["collisional", "photoionization"], loc="upper right"
    )

def plot_branched_luminosity_vs_z(masses=fig_masses, v_rel=0):
    plt.figure(figsize=(7.5, 5.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for M, color in zip(masses, colors):
        coll = solve_redshift_curve(M, v_rel=v_rel, luminosity_func=L_collisional)
        photo = solve_redshift_curve(M, v_rel=v_rel, luminosity_func=L_photoionization)
        plt.loglog(coll["z"], coll["L_over_Ledd"], ":", color=color, linewidth=2, label=mass_labels[M] + " collisional")
        plt.loglog(photo["z"], photo["L_over_Ledd"], "--", color=color, linewidth=2, label=mass_labels[M] + " photoionization")
    setup(r"Luminosity with radiation pressure feedback: branched", r"$L/L_{\rm Edd}$", (1e2, 1e5), (1e-16, 2))
    add_branched_legends(masses, colors)

def plot_single_branch_luminosity_vs_z(masses=fig_masses, v_rel=0):
    plt.figure(figsize=(7.5, 5.2))
    for M in masses:
        sol = solve_redshift_curve(M, v_rel=v_rel)
        plt.loglog(sol["z"], sol["L_over_Ledd"], linewidth=2, label=mass_labels[M])
    setup(r"Luminosity with radiation pressure feedback: single-branch", r"$L/L_{\rm Edd}$", (1e2, 1e5), (1e-16, 2))

def plot_branched_Meff_ratio_vs_z(masses=fig_masses, v_rel=0):
    plt.figure(figsize=(7.5, 5.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for M, color in zip(masses, colors):
        coll = solve_redshift_curve(M, v_rel=v_rel, luminosity_func=L_collisional)
        photo = solve_redshift_curve(M, v_rel=v_rel, luminosity_func=L_photoionization)
        plt.semilogx(coll["z"], coll["M_eff_over_M"], ":", color=color, linewidth=2, label=mass_labels[M] + " collisional")
        plt.semilogx(photo["z"], photo["M_eff_over_M"], "--", color=color, linewidth=2, label=mass_labels[M] + " photoionization")
    plt.axhline(1, linestyle="-", color="black", linewidth=1.2)
    plt.xlabel(r"$z$"); plt.ylabel(r"$M_{\rm eff}/M$"); plt.title(r"Effective mass suppression: branched treatment")
    plt.xlim(1e2, 1e5); plt.ylim(0, 1.05); plt.tight_layout()
    add_branched_legends(masses, colors)

def plot_single_branch_Meff_ratio_vs_z(masses=fig_masses, v_rel=0):
    plt.figure(figsize=(7.5, 5.2))
    for M in masses:
        sol = solve_redshift_curve(M, v_rel=v_rel)
        plt.semilogx(sol["z"], sol["M_eff_over_M"], linewidth=2, label=mass_labels[M])
    plt.axhline(1, linestyle=":", linewidth=1.2)
    plt.xlabel(r"$z$"); plt.ylabel(r"$M_{\rm eff}/M$"); plt.title(r"Effective mass suppression: single-branch treatment")
    plt.xlim(1e2, 1e5); plt.ylim(0, 1.05); plt.legend(); plt.tight_layout()

def main():
    plot_branched_luminosity_vs_z()
    plot_single_branch_luminosity_vs_z()
    plot_single_branch_Meff_ratio_vs_z()
    plot_branched_Meff_ratio_vs_z()
    plt.show()

if __name__ == "__main__":
    main()
