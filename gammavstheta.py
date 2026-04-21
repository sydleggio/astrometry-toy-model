import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lpmv


ELL_MIN = 2
ELL_MAX = 100
EPS = 1e-14


def F_sq(ell):
    """
    |F_l^E|^2 = |F_l^B|^2 = 1 / (N_l^2 * l(l+1))
    N_l^2 = (l+2)(l+1)l(l-1) / 2
    """
    N_sq = ((ell + 2.0) * (ell + 1.0) * ell * (ell - 1.0)) / 2.0
    return 1.0 / (N_sq * ell * (ell + 1.0))


def gamma_parallel_vs_theta(theta_array, ell_min=ELL_MIN, ell_max=ELL_MAX):
    """
    Compute Gamma_o^parallel at each angle in theta_array (radians),
    summing from ell_min to ell_max using lpmv directly.

    G2 = -P_l^1(cos theta) / [l(l+1) sin(theta)]
    At theta=0:  limit = +0.5 for all ell
    At theta=pi: limit = 0
    """
    theta_array = np.asarray(theta_array, dtype=float)
    mu = np.cos(theta_array)
    sin_t = np.sin(theta_array)

    mask_zero = sin_t < 1e-6
    mask_pi   = np.abs(theta_array - np.pi) < 1e-6
    mask_reg  = ~mask_zero & ~mask_pi

    total = np.zeros_like(theta_array)
    for ell in range(ell_min, ell_max + 1):
        ll1 = ell * (ell + 1.0)

        # G1: -1/2 * [ P_l^2(mu) / l(l+1) - P_l(mu) ]
        g1 = -0.5 * (lpmv(2, ell, mu) / ll1 - lpmv(0, ell, mu))

        # G2 with correct limits at theta=0 and theta=pi
        g2 = np.zeros_like(theta_array)
        g2[mask_reg] = -lpmv(1, ell, mu[mask_reg]) / (ll1 * sin_t[mask_reg])
        g2[mask_zero] = 0.5
        g2[mask_pi]   = 0.0

        weight = (2.0 * ell + 1.0) / (4.0 * np.pi) * F_sq(ell)
        total += weight * (g1 + g2)

    return total


def plot_gamma_vs_theta():
    # Theta from just above 0 to just below pi, then append pi=0 by definition
    theta_rad = np.linspace(0.01, np.pi - 0.01, 999)

    print(f'Computing gamma vs theta for ell_max = {ELL_MAX}...')
    gamma = gamma_parallel_vs_theta(theta_rad)

    # Append theta=pi with gamma=0 by definition
    theta_rad = np.append(theta_rad, np.pi)
    gamma = np.append(gamma, 0.0)

    print(f'  gamma(theta->0) = {gamma[0]:.6f}')
    print(f'  gamma(pi/2)     = {gamma[np.argmin(np.abs(theta_rad - np.pi/2))]:.6f}')
    print(f'  gamma(pi)       = {gamma[-1]:.6f}')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(theta_rad, gamma, linewidth=2)
    ax.axhline(0.0, linestyle='--', color='gray', linewidth=1)
    ax.set_xlabel(r'$\Theta$ (radians)')
    ax.set_ylabel(r'$\Gamma_o^\parallel(\Theta)$')
    ax.set_title(rf'Gamma overlap function vs $\Theta$  ($\ell_{{\rm max}}={ELL_MAX}$)')
    ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax.set_xticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_gamma_vs_theta()