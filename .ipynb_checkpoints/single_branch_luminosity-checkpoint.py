'''
Author: Roy Chowdhury
Description: We enforce the self-consistency condition r_ion_end = R, where R is the Stromgren radius, which physically corresponds to an improved
ionization model where the gas first undergoes pure photoionization from r = R_S to r = R and then pure collisional ionization from r = R to r = r_ion.
To achieve this self-consistency, we solve T_S_func(R_func(T_S)) = T_S, where R_func(T_S) is obtained by redefining chi with r_ion_end = R
and R_func(T_S) is given by Eq. 72 in Ali-Haimoud and Kamionkowski (2017). Finally, we plotted luminosity as a function of the self-consistent
temperature at Schwarzchild radius using Eq. 57, which yields a curve that falls between those corresponding to the two limiting cases explored
in the 2017 paper.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from pbh17.constants import *
from pbh17.history import *
from pbh17.bondi import *
from pbh17.utils import *
import pbh17.luminosity as old

E_ion = 13.6 * 1.6e-19

def r_ion(M, z, v_rel=0):
    Tau, T_inf, r_b = tau(M, z, v_rel), T_b(z), r_B(M, z, v_rel)
    T_ion = 1.5e4
    return Tau * (T_inf/T_ion) * r_b
    
def chi(R, M, z, v_rel=0):
    r_is = r_ion(M, z, v_rel)
    xe = x_e_pbh(z)
    r_coll = r_is * (1+xe)**8 / 256
    if R >= r_is:
        return 1
    elif R < r_coll:
        return (2/(1+xe))**12
    return (r_is/R)**(3/2)

def Y_S(R, M, z, v_rel=0):
    xe, ta, ch = x_e_pbh(z), tau(M, z, v_rel), chi(R, M, z, v_rel)
    return ch**(-2/3) * 2/(1+xe) * ta/4 * (1 - 2.5*ta)**(1/3) * m_p/m_e

def F(Y):
    return Y * (1 + Y / 0.27)**(-1/3)

def T_S_func(R, M, z, v_rel=0):
    return m_e * c**2 / k * F(Y_S(R, M, z, v_rel))

def J(X):
    X = np.asarray(X); out = np.empty_like(X, dtype=float); low = X < 1
    out[low] = 4/pi * np.sqrt(2/(pi * X[low])) * (1 + 5.5 * X[low]**1.25)
    out[~low] = 27/(2*pi) * (np.log(2 * X[~low] * np.exp(-np.euler_gamma) + 0.08) + 4/3)
    return out

def R_func(TS, M, z, v_rel=0):
    X = k * TS / (m_e * c**2)
    return 2e-4 * r_ion(M, z, v_rel) * (J(X) * np.log(k * TS / E_ion))**1.16

def temperature_residual(log_TS, M, z, v_rel=0):
    TS_guess = np.exp(log_TS)
    TS_new = T_S_func(R_func(TS_guess, M, z, v_rel), M, z, v_rel)
    return np.log(TS_new / TS_guess)

def solve_TS(M, z, v_rel=0):
    Tc = old.T_S(M, z, "collisional", v_rel)
    Tp = old.T_S(M, z, "photoionization", v_rel)
    lo, hi = np.log(min(Tc, Tp)), np.log(max(Tc, Tp))
    flo = temperature_residual(lo, M, z, v_rel)
    fhi = temperature_residual(hi, M, z, v_rel)
    if np.isclose(flo, 0, atol=1e-12):
        return np.exp(lo)
    if np.isclose(fhi, 0, atol=1e-12):
        return np.exp(hi)
    log_TS = brentq(temperature_residual, lo, hi, args=(M, z, v_rel))
    return np.exp(log_TS)

def L_from_TS(TS, M, z, v_rel=0):
    Mdot = 4 * pi * rho_b(z) * r_B(M, z, v_rel)**2 * v_B(z, v_rel) * lambda_acc(M, z, v_rel)
    L_Edd = 4 * pi * G * M * M_sun * m_p * c / sigma_T
    X = k * TS / (m_e * c**2)
    return alpha * k * TS / (m_p * c**2) * J(X) * (Mdot * c**2)**2 / L_Edd

def L(M, z, v_rel=0, R=None):
    TS = solve_TS(M, z, v_rel) if R is None else T_S_func(R, M, z, v_rel)
    return L_from_TS(TS, M, z, v_rel)

def solve_redshift_curve(M, z_grid=None, v_rel=0):
    z = np.logspace(np.log10(30), 5, 160) if z_grid is None else np.asarray(z_grid)
    TS = np.array([solve_TS(M, zi, v_rel) for zi in z])
    TS_coll = old.T_S(M, z, "collisional", v_rel)
    TS_photo = old.T_S(M, z, "photoionization", v_rel)
    L_single = L_from_TS(TS, M, z, v_rel)
    L_coll = L_from_TS(TS_coll, M, z, v_rel)
    L_photo = L_from_TS(TS_photo, M, z, v_rel)
    Ledd = old.L_Edd(M)
    return {"z": z, "T_S": TS, "L": L_single, "L_over_L_Edd": L_single/Ledd, "L_coll_over_L_Edd": L_coll/Ledd, "L_photo_over_L_Edd": L_photo/Ledd}

def plot_single_branch_luminosity(masses=(1, 1e2, 1e4, 1e5), z_grid=None, v_rel=0):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for M, color in zip(masses, colors):
        s = solve_redshift_curve(M, z_grid, v_rel)
        z, L = s["z"], s["L_over_L_Edd"]
        Lc, Lp = s["L_coll_over_L_Edd"], s["L_photo_over_L_Edd"]
        plt.loglog(z, Lc, ":", color=color, label=mass_labels[M]+" collisional")
        plt.loglog(z, L, "-", color=color, label=mass_labels[M]+" combined")
        plt.loglog(z, Lp, "--", color=color, label=mass_labels[M]+" photo")
    plt.xlabel(r"$z$")
    plt.ylabel(r"$L/L_{\rm Edd}$")
    plt.xlim(30, 1e5)
    plt.legend(fontsize=8)
    plt.tight_layout()
    return plt.gcf()

if __name__ == "__main__":
    plot_single_branch_luminosity()
    plt.show()
