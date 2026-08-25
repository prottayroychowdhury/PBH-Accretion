"""Mass enclosed by the Hubble-limited radial inflow profile."""

import numpy as np
import matplotlib.pyplot as plt

from pbh17.constants import G, M_sun, c, pi
from pbh17.history import Hubble, rho_b


fig_masses = (1, 1e2, 1e4, 1e5, 1e6)
z_vals = np.logspace(2, 5, 300)


def R_S(M):
    """Schwarzschild radius for a PBH mass M given in solar masses."""
    return 2 * G * M * M_sun / c**2


def r_I(M, z):
    """Influence radius r_I = (GM/H^2)^(1/3)."""
    return (G * M * M_sun / Hubble(z)**2)**(1/3)


def M_acc(M, z):
    r"""Evaluate

    M_acc = integral[R_S, r_I] (4 pi rho_b / 5H) sqrt(2GMr) dr.

    The radial integral is evaluated analytically.
    """
    M_kg = M * M_sun
    r_min = R_S(M)
    r_max = r_I(M, z)
    radial_integral = (2/3) * np.sqrt(2 * G * M_kg) * (
        r_max**(3/2) - r_min**(3/2)
    )
    return (4 * pi * rho_b(z) / (5 * Hubble(z))) * radial_integral


def plot_accreted_mass_fraction(masses=fig_masses, z_grid=z_vals):
    plt.figure(figsize=(7.5, 5.2))
    for M in masses:
        fraction = M_acc(M, z_grid) / (M * M_sun)
        exponent = int(np.log10(M))
        plt.loglog(z_grid, fraction, linewidth=2, label=rf"$10^{{{exponent}}}\,M_\odot$")

    plt.xlabel(r"$z$")
    plt.ylabel(r"$M_{\rm acc}/M$")
    plt.title(r"Mass within the influence radius")
    plt.xlim(z_grid.min(), z_grid.max())
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


if __name__ == "__main__":
    plot_accreted_mass_fraction()
    plt.show()
