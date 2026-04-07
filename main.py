import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lpmv


# ============================================================
#                        CONSTANTS
# ============================================================

# Assuming same noise for every star of 1 mas (this is in radians)
sigma_rad = 4.8481e-9    
# 30 minute cadence of Kepler, converted to seconds     
dt_seconds = 30.0 * 60.0       
# Total observation time of 3.5 years, converted to seconds
T_obs_seconds = 3.5 * 365.25 * 24.0 * 3600.0  

# Low frequency cutoff is 1/T, high frequency cutoff is Nyquist = 1/(2*dt)
f_l = 1.0 / T_obs_seconds
f_h = 1.0 / (2.0 * dt_seconds)

# Noise PSD convention from Romano PTA paper
P_n = 2.0 * sigma_rad**2 * dt_seconds

# Reference frequency from Romano: 1/year converted to seconds
f_yr = 1.0 / (365.25 * 24.0 * 3600.0)

# Possible spot for tuning the overall amplitude of the GW signal?
A_gw = 1.0

# Setting summation limit as 30 for now, can be increased 
ELL_MAX = 30
EPS = 1e-14

# Grid / field
FIELD_SIZE_DEG = 10.0
N_STARS = 100

# Put exact coordinates here later as an (N, 2) array in degrees, like STAR_COORDS_DEG = np.array([[0.1, -0.2], [1.4, 3.2], ...])
# Leave as NONE to generate random positions uniformly in a square field of size FIELD_SIZE_DEG x FIELD_SIZE_DEG
STAR_COORDS_DEG = None
RANDOM_SEED = 1234


# ============================================================
#                        STAR POSITIONS
# ============================================================
def build_star_positions(
    star_coords_deg=None,
    n_stars=N_STARS,
    field_size_deg=FIELD_SIZE_DEG,
    seed=RANDOM_SEED,
):
    if star_coords_deg is not None:
        stars = np.asarray(star_coords_deg, dtype=float)
        if stars.ndim != 2 or stars.shape[1] != 2:
            raise ValueError('star_coords_deg must have shape (N, 2).')
        return stars

    rng = np.random.default_rng(seed)
    half = field_size_deg / 2.0
    stars = rng.uniform(-half, half, size=(n_stars, 2))
    return stars


def pairwise_theta_flat_sky(stars_deg):
    '''
    Flat-sky angular separations in radians.
    '''
    stars_rad = np.deg2rad(stars_deg)
    dx = stars_rad[:, 0][:, None] - stars_rad[:, 0][None, :]
    dy = stars_rad[:, 1][:, None] - stars_rad[:, 1][None, :]
    theta = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(theta, np.nan)
    return theta

# ============================================================
#                     DEFINING EQUATIONS
# ============================================================

# Defining the Kernel functions G_l^(1) and G_l^(2) using associated Legendre polynomials

def G1(ell, theta):
    '''
    G_l^(1)(Theta) = -1/2 [ P_l^2(cos Theta)/(l(l+1)) - P_l(cos Theta) ]
    '''
    theta = np.asarray(theta, dtype=float)
    mu = np.cos(theta)
    return -0.5 * (lpmv(2, ell, mu) / (ell * (ell + 1.0)) - lpmv(0, ell, mu))


def G2(ell, theta):
    '''
    G_l^(2)(Theta) = -1/[l(l+1)] * P_l^1(cos Theta)/sin(Theta)

    Finite limit at Theta -> 0 is 1/2.
    '''
    theta = np.asarray(theta, dtype=float)
    mu = np.cos(theta)
    sin_t = np.sin(theta)
    # Need to handle the singularity at Theta = 0 carefully so there is no undefined term
    out = np.zeros_like(theta, dtype=float)
    mask = np.abs(sin_t) > 1e-12
    out[mask] = -(lpmv(1, ell, mu[mask]) / sin_t[mask]) / (ell * (ell + 1.0))
    out[~mask] = 0.5
    return out


def N_l_sq(ell):
    '''
    N_l^2 = (l+2)! / [2 (l-2)!] = ((l+2)(l+1)l(l-1))/2
    '''
    if ell < 2:
        raise ValueError('ell must be >= 2')
    return ((ell + 2.0) * (ell + 1.0) * ell * (ell - 1.0)) / 2.0


def F_sq(ell):
    '''
    |F_l^{E,TE}|^2 = |F_l^{B,TB}|^2
    = 1 / [N_l^2 * l(l+1)]
    '''
    return 1.0 / (N_l_sq(ell) * ell * (ell + 1.0))


# ============================================================
#                 GAMMA OVERLAP FUNCTION  
# ============================================================
def gamma_parallel(theta, ell_max=ELL_MAX):
    theta = np.asarray(theta, dtype=float)
    total = np.zeros_like(theta, dtype=float)
    for ell in range(2, ell_max + 1):
        weight = (2.0 * ell + 1.0) / (4.0 * np.pi) * F_sq(ell)
        total += weight * (G1(ell, theta) + G2(ell, theta))

    return total


def sum_inverse_gamma(theta_matrix):
    '''
    Sum over unique pairs i < j of 1 / Gamma_o^parallel(theta_ij).
    '''
    gamma = gamma_parallel(theta_matrix)
    iu = np.triu_indices_from(gamma, k=1)
    vals = gamma[iu]
    vals = vals[np.isfinite(vals) & (np.abs(vals) > EPS)]
    return np.sum(1.0 / np.abs(vals))


# ============================================================
#              COMMON PROCESS SNR CALCULATIONS
# ============================================================
def rho_cp_sq_weak_from_ratio(ratio):
    '''
    Weak low-frequency branch:
        rho_cp^2 = 100 * P_gw^2(f_l) * f_l^(7/3) / sigma^2
    '''
    ratio = np.asarray(ratio, dtype=float)
    P_gw = ratio * P_n
    return 100.0 * (P_gw ** 2) * (f_l ** (7.0 / 3.0)) / (sigma_rad ** 2)


def rho_cp_sq_intermediate(sum_inv_gamma):
    '''
    Intermediate low-frequency branch:
        rho_cp^2 = [Abar_gw^2 * f_l^(7/3)] / [192 pi^3] * sum(1/Gamma)

    with
        Abar_gw^2 = [A_gw^2 / (12 pi^2)] * f_yr^(4/3)
    '''
    Abar_sq = (A_gw ** 2) / (12.0 * np.pi ** 2) * (f_yr ** (4.0 / 3.0))
    return (Abar_sq * (f_l ** (7.0 / 3.0)) / (192.0 * np.pi ** 3)) * sum_inv_gamma


def piecewise_rho_cp(ratio, sum_inv_gamma):
    # Want to stitch together the weak and intermediate branches to get the full curve 
    # The transition is where they are equal, which gives us a critical ratio to compare against
    rho_sq_weak = rho_cp_sq_weak_from_ratio(ratio)
    rho_sq_int = rho_cp_sq_intermediate(sum_inv_gamma)

    rho_weak = np.sqrt(np.maximum(rho_sq_weak, 0.0))
    rho_int = np.sqrt(np.maximum(rho_sq_int, 0.0))

    # Solve weak(ratio) = intermediate plateau for the crossover.
    # weak: rho^2 = 100 * (ratio^2 * P_n^2) * f_l^(7/3) / sigma^2
    # => rho = ratio * sqrt(100 * P_n^2 * f_l^(7/3) / sigma^2)
    weak_slope = np.sqrt(100.0 * (P_n ** 2) * (f_l ** (7.0 / 3.0)) / (sigma_rad ** 2))
    transition_ratio = rho_int / weak_slope if weak_slope > 0 else np.inf

    rho = np.where(ratio <= transition_ratio, rho_weak, rho_int)
    return rho, rho_int, transition_ratio


# ============================================================
#                MAIN FUNCTION - OUTPUTS AND PLOT
# ============================================================
def main():
    stars_deg = build_star_positions(STAR_COORDS_DEG)
    theta = pairwise_theta_flat_sky(stars_deg)
    sum_inv = sum_inverse_gamma(theta)

    rho_int = np.sqrt(np.maximum(rho_cp_sq_intermediate(sum_inv), 0.0))
    weak_slope = np.sqrt(100.0 * (P_n ** 2) * (f_l ** (7.0 / 3.0)) / (sigma_rad ** 2))
    transition_ratio = rho_int / weak_slope if weak_slope > 0 else 1.0

    x_max = max(5.0 * transition_ratio, 1e-4)
    ratio = np.logspace(-4, 1, 5000)  
    rho_cp, plateau, transition = piecewise_rho_cp(ratio, sum_inv)

    print('\nINPUT PARAMETERS:')
    print(f'dt = {dt_seconds:.3f} s')
    print(f'T_obs = {T_obs_seconds:.3e} s')
    print(f'f_low = {f_l:.6e} Hz')
    print(f'f_high = {f_h:.6e} Hz')
    print(f'P_n = {P_n:.6e} rad^2 s')
    print(f'Number of stars = {len(stars_deg)}')
    print(f'Sum(1/Gamma) = {sum_inv:.6e}')
    print(f'Plateau rho_cp = {plateau:.6e}')
    print(f'Transition P_gw/P_n = {transition:.6e}')

    fig, ax = plt.subplots()
    ax.plot(ratio, rho_cp, linewidth=2)
    ax.axvline(transition, linestyle='--', linewidth=1)
    ax.axhline(plateau, linestyle=':', linewidth=1)
    ax.semilogx(ratio, rho_cp, antialiased=True, linewidth=2.5)
    ax.set_xlabel(r'$P_{gw}/P_n$')
    ax.set_ylabel(r'$\rho_{cp}$')
    ax.set_title('Low-Frequency Common Process SNR')
    ax.grid(alpha=0.3)
    ax.set_xlim(0.0, x_max)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()