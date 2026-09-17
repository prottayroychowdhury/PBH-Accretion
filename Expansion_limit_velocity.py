import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pbh17 import *

z_ref = 11.5
z_init = 8000
H_ref = float(Hubble(z_ref))
a_init = (1 + z_ref) / (1 + z_init)
points = 5000
masses = np.array([1e4, 1e5, 1e6])
x_plot = 10

def background(N):
    a = np.exp(N)
    z = (1 + z_ref) / a - 1
    Hhat = Hubble(z) / H_ref
    beta = 4 * x_e(z) * sigma_T * rho_cmb(z) / (3 * m_p * c * H_ref)
    return Hhat, beta

def rhs(N, F):
    Hhat, beta = background(N)
    return [np.exp(-2*N)/Hhat - (1 + beta/Hhat)*F[0]]

def solve_velocity():
    n = np.linspace(np.log(a_init), 0, 5000)
    solution = solve_ivp(rhs, (n[0], n[-1]), [0.], t_eval=n, rtol = 1e-9, atol = 1e-12)
    return np.exp(n), solution.y[0]

def velocity(a, x, a_grid, F_grid):
    a, x = np.asarray(a), np.asarray(x)
    return -np.interp(np.log(a), np.log(a_grid), F_grid) / x**2

def plot_velocity(z, F, v_kms):
    plt.figure()
    plt.plot(z, F)
    plt.xscale('log')
    plt.xlim(z_ref, z_init)
    plt.xlabel('z')
    plt.ylabel(r'$F(z)=-x^2u(z,x)$')
    plt.title('Outer velocity: direct numerical solution')

    plt.figure()
    for mass, v in zip(masses, v_kms.T):
        plt.plot(z, -v, label=rf'$10^{{{int(np.log10(mass))}}}\,M_\odot$')
    plt.xscale('log')
    plt.xlim(z_ref, z_init)
    plt.xlabel('z')
    plt.ylabel(r'Inward peculiar speed $-v$ [km/s]')
    plt.title(f'Physical velocity at fixed comoving x={x_plot:g}')
    plt.legend()

def main():
    a, F = solve_velocity()
    z = (1 + z_ref) / a - 1
    r_I0 = (G * masses * M_sun / H_ref**2)**(1/3)
    v_E = H_ref * r_I0
    u = -F / x_plot**2
    v_kms = u[:, None] * v_E / 1000
    plot_velocity(z, F, v_kms)
    plt.show()

if __name__ == '__main__':
    main()
 