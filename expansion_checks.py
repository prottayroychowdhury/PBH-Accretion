import numpy as np
import matplotlib.pyplot as plt
from pbh17 import *

expansion_masses = (1e4, 1e5, 1e6)

def plot_tB_H(masses=mass_labels.keys()):
    z = np.logspace(0, 11, 1000)
    plt.figure(figsize=(7.2, 5.0))
    for M in masses:
        plt.loglog(z, t_B(M, z) * Hubble(z), label=mass_labels[M])
    plt.axhline(1, linestyle=":", linewidth=1, label=r"$t_BH=1$")
    plt.xlabel(r"$z$"); plt.ylabel(r"$t_BH$");
    plt.title(r"Bondi timescale vs expansion timescale")
    plt.xlim(1, 1e11); plt.ylim(1e-6, 1e4); plt.legend(); plt.tight_layout()

def r_I(M, z):
    return (G * M * M_sun / Hubble(z)**2)**(1/3)

def M_enc_over_M(M, z):
    R_S = 2 * G * M * M_sun / c**2
    correction = 1 - (R_S / r_I(M, z))**(3/2)
    return (8 * pi * np.sqrt(2) / 15) * G * rho_b(z) / Hubble(z)**2 * correction

def plot_M_enc_over_M(masses=expansion_masses):
    z = np.logspace(0, 11, 1000)
    plt.figure(figsize=(7.2, 5.0))
    for M in masses:
        fraction = M_enc_over_M(M, z)
        valid = fraction > 0
        exponent = int(np.log10(M))
        plt.loglog(z[valid], fraction[valid], label=rf"$10^{{{exponent}}}\,M_\odot$")
    plt.xlabel(r"$z$"); plt.ylabel(r"$M_{\rm enc}/M$")
    plt.title(r"Self-gravity in the expansion-dominated regime")
    plt.xlim(1, 1e11); plt.ylim(1e-8, 1)
    plt.legend(); plt.tight_layout()

def main():
    plot_tB_H()
    plot_M_enc_over_M()
    plt.show()

if __name__ == "__main__":
    main()
