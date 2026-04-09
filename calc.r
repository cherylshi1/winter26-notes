"""
Polarization of Light - Data Analysis
PHY294 Lab Report
Cheryl Shi and Andy Wen

This script performs curve fitting and plotting for all three exercises:
  Exercise 1: Two Polarizers (Malus' Law verification)
  Exercise 2: Three Polarizers (sequential polarization)
  Exercise 3: Brewster's Angle (polarization by reflection)

Required data files (place in same directory):
  trial3.txt       - Exercise 1 data
  2trial1.txt      - Exercise 2 data
  1realtrial4.txt  - Exercise 3, no polarizer
  andyfirst1.txt   - Exercise 3, parallel (horizontal) polarizer
  vertical2.txt    - Exercise 3, perpendicular (vertical) polarizer
"""
pip install numpy scipy matplotlib
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from math import floor, log10

# ============================================================
# Uncertainties
# ============================================================
sigma_I = 0.01          # V, light sensor display resolution
sigma_theta_deg = 0.5   # degrees, rotary motion sensor resolution
sigma_theta = sigma_theta_deg * np.pi / 180  # radians

def round_unc(val, unc):
    """Round uncertainty to 1 sig fig, match value digits."""
    if unc == 0:
        return val, unc
    mag = floor(log10(abs(unc)))
    unc_r = round(unc, -mag)
    val_r = round(val, -mag)
    return val_r, unc_r


# ============================================================
# EXERCISE 1: Two Polarizers — Verification of Malus' Law
# ============================================================
print("=" * 60)
print("EXERCISE 1: Two Polarizers")
print("=" * 60)

data1 = np.loadtxt('trial3.txt', skiprows=2)
theta1_raw = data1[:, 0]   # radians, 0 to ~ -3.16
I1 = data1[:, 1]           # volts

# Model: I(theta) = I0 * cos^2(theta - theta_0) + I_bg
# theta_0 accounts for small misalignment between the initial polarizer setting and true zero
def malus_law(theta, I0, theta0, I_bg):
    return I0 * np.cos(theta - theta0)**2 + I_bg

popt1, pcov1 = curve_fit(
    malus_law, theta1_raw, I1,
    p0=[4.0, 0.0, 0.0],
    sigma=sigma_I * np.ones(len(I1)),
    absolute_sigma=True
)
perr1 = np.sqrt(np.diag(pcov1))

I0_fit    = popt1[0]
th0_fit   = popt1[1]
Ibg_fit   = popt1[2]
I0_err    = perr1[0]
th0_err   = perr1[1]
Ibg_err   = perr1[2]

I1_pred = malus_law(theta1_raw, *popt1)
res1    = I1 - I1_pred
chi2_1  = np.sum((res1 / sigma_I)**2)
dof1    = len(I1) - 3
chi2r_1 = chi2_1 / dof1
R2_1    = 1 - np.sum(res1**2) / np.sum((I1 - np.mean(I1))**2)

print(f"Data points: {len(I1)}")
print(f"I0     = {I0_fit:.3f} +/- {I0_err:.3f} V")
print(f"theta0 = {np.degrees(th0_fit):.2f} +/- {np.degrees(th0_err):.2f} deg")
print(f"I_bg   = {Ibg_fit:.4f} +/- {Ibg_err:.4f} V")
print(f"R^2    = {R2_1:.6f}")
print(f"Reduced chi^2 = {chi2r_1:.1f} (dof = {dof1})")
print()

# --- Figure 1: I vs theta with Malus' Law fit ---
fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(8, 7),
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   sharex=True)

theta_eff     = theta1_raw - th0_fit
theta_eff_deg = np.degrees(theta_eff)

theta_plot = np.linspace(theta1_raw.min(), theta1_raw.max(), 500)
I_plot     = malus_law(theta_plot, *popt1)
theta_plot_eff_deg = np.degrees(theta_plot - th0_fit)

ax1a.errorbar(theta_eff_deg, I1, yerr=sigma_I, xerr=sigma_theta_deg,
              fmt='o', markersize=2, color='steelblue', alpha=0.5,
              ecolor='gray', elinewidth=0.5, capsize=0, label='Data')
ax1a.plot(theta_plot_eff_deg, I_plot, 'r-', linewidth=1.5,
          label=r'Fit: $I = I_0 \cos^2(\theta - \theta_0) + I_{bg}$')
ax1a.set_ylabel('Intensity (V)')
ax1a.legend(fontsize=10)
ax1a.set_title("Exercise 1: Verification of Malus' Law (Two Polarizers)")

ax1b.errorbar(theta_eff_deg, res1, yerr=sigma_I,
              fmt='o', markersize=2, color='steelblue', alpha=0.5,
              ecolor='gray', elinewidth=0.5, capsize=0)
ax1b.axhline(0, color='red', linewidth=1)
ax1b.set_xlabel(r'Analyzer Angle $\theta - \theta_0$ (degrees)')
ax1b.set_ylabel('Residual (V)')

fig1.tight_layout()
fig1.savefig('exercise1_malus_law.png', dpi=200, bbox_inches='tight')
plt.close(fig1)
print("Saved: exercise1_malus_law.png")

# --- Figure 2: Linearized — I vs cos^2(theta - theta_0) ---
fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(8, 7),
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   sharex=True)

cos2_eff = np.cos(theta1_raw - th0_fit)**2

def linear_model(x, m, b):
    return m * x + b

popt_lin, pcov_lin = curve_fit(linear_model, cos2_eff, I1,
                               sigma=sigma_I * np.ones(len(I1)),
                               absolute_sigma=True)
perr_lin = np.sqrt(np.diag(pcov_lin))

I_lin_pred = linear_model(cos2_eff, *popt_lin)
res_lin    = I1 - I_lin_pred
chi2_lin   = np.sum((res_lin / sigma_I)**2)
dof_lin    = len(I1) - 2
chi2r_lin  = chi2_lin / dof_lin
R2_lin     = 1 - np.sum(res_lin**2) / np.sum((I1 - np.mean(I1))**2)

print(f"\nLinearized fit: I = m * cos^2(theta - theta_0) + b")
print(f"m (= I0)   = {popt_lin[0]:.3f} +/- {perr_lin[0]:.3f} V")
print(f"b (= I_bg) = {popt_lin[1]:.4f} +/- {perr_lin[1]:.4f} V")
print(f"R^2 = {R2_lin:.6f}")
print(f"Reduced chi^2 = {chi2r_lin:.1f} (dof = {dof_lin})")

x_smooth = np.linspace(0, 1, 100)
sigma_cos2 = np.abs(-2 * np.cos(theta1_raw - th0_fit) *
                     np.sin(theta1_raw - th0_fit)) * sigma_theta

ax2a.errorbar(cos2_eff, I1, yerr=sigma_I, xerr=sigma_cos2,
              fmt='o', markersize=2, color='steelblue', alpha=0.5,
              ecolor='gray', elinewidth=0.5, capsize=0, label='Data')
ax2a.plot(x_smooth, linear_model(x_smooth, *popt_lin),
          'r-', linewidth=1.5,
          label=r'Linear fit: $I = I_0 \cos^2\theta + I_{bg}$')
ax2a.set_ylabel('Intensity (V)')
ax2a.legend(fontsize=10)
ax2a.set_title(r"Exercise 1: Linearized Malus' Law ($I$ vs $\cos^2\theta$)")

ax2b.errorbar(cos2_eff, res_lin, yerr=sigma_I,
              fmt='o', markersize=2, color='steelblue', alpha=0.5,
              ecolor='gray', elinewidth=0.5, capsize=0)
ax2b.axhline(0, color='red', linewidth=1)
ax2b.set_xlabel(r'$\cos^2(\theta - \theta_0)$')
ax2b.set_ylabel('Residual (V)')

fig2.tight_layout()
fig2.savefig('exercise1_linearized.png', dpi=200, bbox_inches='tight')
plt.close(fig2)
print("Saved: exercise1_linearized.png\n")


# ============================================================
# EXERCISE 2: Three Polarizers
# ============================================================
print("=" * 60)
print("EXERCISE 2: Three Polarizers")
print("=" * 60)

data2 = np.loadtxt('2trial1.txt', skiprows=2)
theta2_raw = data2[:, 0]   # radians, 0 to ~ -3.15
I2 = data2[:, 1]

# The measured angle Theta is the rotation of the middle polarizer.
# Physical angle phi = Theta + phi_0
# Theory: I3 = (I1/4) * sin^2(2*phi) + background

def three_pol_model(theta, A, phi0, I_bg):
    phi = theta + phi0
    return A * np.sin(2 * phi)**2 + I_bg

popt2, pcov2 = curve_fit(
    three_pol_model, theta2_raw, I2,
    p0=[0.75, 0.05, 0.0],
    sigma=sigma_I * np.ones(len(I2)),
    absolute_sigma=True
)
perr2 = np.sqrt(np.diag(pcov2))

A_fit     = popt2[0]
phi0_fit  = popt2[1]
Ibg2_fit  = popt2[2]
A_err     = perr2[0]
phi0_err  = perr2[1]
Ibg2_err  = perr2[2]

I2_pred = three_pol_model(theta2_raw, *popt2)
res2    = I2 - I2_pred
chi2_2  = np.sum((res2 / sigma_I)**2)
dof2    = len(I2) - 3
chi2r_2 = chi2_2 / dof2
R2_2    = 1 - np.sum(res2**2) / np.sum((I2 - np.mean(I2))**2)

I1_calc     = 4 * A_fit
I1_calc_err = 4 * A_err

print(f"Data points: {len(I2)}")
print(f"A (= I1/4)  = {A_fit:.4f} +/- {A_err:.4f} V")
print(f"I1 = 4A     = {I1_calc:.3f} +/- {I1_calc_err:.3f} V")
print(f"phi_0       = {np.degrees(phi0_fit):.1f} +/- {np.degrees(phi0_err):.1f} deg")
print(f"I_bg        = {Ibg2_fit:.5f} +/- {Ibg2_err:.5f} V")
print(f"R^2         = {R2_2:.6f}")
print(f"Reduced chi^2 = {chi2r_2:.2f} (dof = {dof2})")
print()

# --- Figure 3: Three polarizer I vs phi ---
fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(8, 7),
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   sharex=True)

phi_data_deg = np.degrees(theta2_raw + phi0_fit)

theta_plot2  = np.linspace(theta2_raw.min(), theta2_raw.max(), 500)
phi_plot_deg = np.degrees(theta_plot2 + phi0_fit)
I_plot2      = three_pol_model(theta_plot2, *popt2)

ax3a.errorbar(phi_data_deg, I2, yerr=sigma_I, xerr=sigma_theta_deg,
              fmt='o', markersize=2, color='steelblue', alpha=0.5,
              ecolor='gray', elinewidth=0.5, capsize=0, label='Data')
ax3a.plot(phi_plot_deg, I_plot2, 'r-', linewidth=1.5,
          label=r'Fit: $I = A\,\sin^2(2\varphi) + I_{bg}$')
ax3a.set_ylabel('Intensity (V)')
ax3a.legend(fontsize=10)
ax3a.set_title(r'Exercise 2: Three Polarizers ($I$ vs $\varphi$)')

ax3b.errorbar(phi_data_deg, res2, yerr=sigma_I,
              fmt='o', markersize=2, color='steelblue', alpha=0.5,
              ecolor='gray', elinewidth=0.5, capsize=0)
ax3b.axhline(0, color='red', linewidth=1)
ax3b.set_xlabel(r'Physical Angle $\varphi$ (degrees)')
ax3b.set_ylabel('Residual (V)')

fig3.tight_layout()
fig3.savefig('exercise2_three_polarizers.png', dpi=200, bbox_inches='tight')
plt.close(fig3)
print("Saved: exercise2_three_polarizers.png\n")


# ============================================================
# EXERCISE 3: Brewster's Angle
# ============================================================
print("=" * 60)
print("EXERCISE 3: Brewster's Angle")
print("=" * 60)

# The sensor position (degrees) relates to incidence angle by: theta_i = sensor / 2
data_nopol = np.loadtxt('1realtrial4.txt', skiprows=2)
theta_nopol = data_nopol[:, 0] / 2
I_nopol     = data_nopol[:, 1]

data_par = np.loadtxt('andyfirst1.txt', skiprows=2)
theta_par = data_par[:, 0] / 2
I_par     = data_par[:, 1]

data_perp = np.loadtxt('vertical2.txt', skiprows=2)
theta_perp = data_perp[:, 0] / 2
I_perp     = data_perp[:, 1]

# Uncertainty in incidence angle: instrument reads sensor position to +/-0.5 deg
# so theta_i uncertainty = 0.5/2 = 0.25 degrees
sigma_theta_brewster = 0.25  # degrees

# --- Bin data ---
def bin_data(theta, I, bin_width=0.5):
    bins = np.arange(theta.min(), theta.max() + bin_width, bin_width)
    centers, means, errs = [], [], []
    for i in range(len(bins) - 1):
        mask = (theta >= bins[i]) & (theta < bins[i + 1])
        n = np.sum(mask)
        if n >= 2:
            centers.append((bins[i] + bins[i + 1]) / 2)
            means.append(np.mean(I[mask]))
            errs.append(np.std(I[mask]) / np.sqrt(n))
    return np.array(centers), np.array(means), np.array(errs)

theta_np_b, I_np_b, err_np_b = bin_data(theta_nopol, I_nopol, 0.5)
theta_pa_b, I_pa_b, err_pa_b = bin_data(theta_par,   I_par,   0.5)
theta_pe_b, I_pe_b, err_pe_b = bin_data(theta_perp,  I_perp,  0.5)

# --- Determine Brewster angle via parabolic fit near minimum ---
idx_min = np.argmin(I_np_b)
theta_min_approx = theta_np_b[idx_min]

theta_B_list = []
for delta in [3, 4, 5, 6, 7]:
    m = (theta_np_b >= theta_min_approx - delta) & \
        (theta_np_b <= theta_min_approx + delta)
    if np.sum(m) >= 3:
        p = np.polyfit(theta_np_b[m], I_np_b[m], 2)
        theta_B_list.append(-p[1] / (2 * p[0]))

theta_B   = np.mean(theta_B_list)
delta_thB = max(np.std(theta_B_list), 1.0)

# Round
theta_B_r, delta_thB_r = round_unc(theta_B, delta_thB)

print(f"\nBrewster angle: theta_B = {theta_B_r} +/- {delta_thB_r} degrees")

# Refractive index
n_exp   = np.tan(np.radians(theta_B))
dn_exp  = (1 / np.cos(np.radians(theta_B)))**2 * np.radians(delta_thB)
n_r, dn_r = round_unc(n_exp, dn_exp)

print(f"n = tan(theta_B) = {n_r} +/- {dn_r}")
print(f"Accepted value for acrylic: ~1.49")
print(f"Percent difference: {abs(n_exp - 1.49)/1.49 * 100:.1f}%")
print()

# --- Figure 4: Reflected intensity vs incidence angle ---
fig4, ax4 = plt.subplots(figsize=(9, 6))

ax4.errorbar(theta_np_b, I_np_b, yerr=err_np_b, xerr=sigma_theta_brewster,
             fmt='s-', markersize=3, color='black', linewidth=1,
             elinewidth=0.5, capsize=2, label='No polarizer (total)', alpha=0.8)
ax4.errorbar(theta_pa_b, I_pa_b, yerr=err_pa_b, xerr=sigma_theta_brewster,
             fmt='o-', markersize=3, color='crimson', linewidth=1,
             elinewidth=0.5, capsize=2,
             label=r'Parallel ($I_\parallel$)', alpha=0.8)
ax4.errorbar(theta_pe_b, I_pe_b, yerr=err_pe_b, xerr=sigma_theta_brewster,
             fmt='^-', markersize=3, color='royalblue', linewidth=1,
             elinewidth=0.5, capsize=2,
             label=r'Perpendicular ($I_\perp$)', alpha=0.8)

ax4.axvline(theta_B, color='green', linestyle='--', linewidth=1.2,
            label=f'Brewster angle = {theta_B:.1f}' + r'$^\circ$')

ax4.set_xlabel(r'Angle of Incidence $\theta_i$ (degrees)')
ax4.set_ylabel('Reflected Intensity (V)')
ax4.set_title('Exercise 3: Reflected Intensity vs Incidence Angle')
ax4.legend(fontsize=9)

fig4.tight_layout()
fig4.savefig('exercise3_brewster_intensity.png', dpi=200, bbox_inches='tight')
plt.close(fig4)
print("Saved: exercise3_brewster_intensity.png")

# --- Figure 5: Theoretical Fresnel reflectances ---
fig5, ax5 = plt.subplots(figsize=(8, 5))

n1 = 1.00
n2 = n_exp

theta_th  = np.linspace(0, 89.9, 500)
theta_rad = np.radians(theta_th)

sin_t2 = (n1 / n2) * np.sin(theta_rad)
sin_t2 = np.clip(sin_t2, -1, 1)
cos_t2 = np.sqrt(1 - sin_t2**2)
cos_t1 = np.cos(theta_rad)

Rs = ((n1 * cos_t1 - n2 * cos_t2) / (n1 * cos_t1 + n2 * cos_t2))**2
Rp = ((n2 * cos_t1 - n1 * cos_t2) / (n2 * cos_t1 + n1 * cos_t2))**2

ax5.plot(theta_th, Rs, 'b-', linewidth=2,
         label=r'$R_\perp$ (s-polarization)')
ax5.plot(theta_th, Rp, 'r-', linewidth=2,
         label=r'$R_\parallel$ (p-polarization)')
ax5.axvline(theta_B, color='green', linestyle='--', linewidth=1.2,
            label=rf'$\theta_B$ = {theta_B:.1f}' + r'$^\circ$')

ax5.set_xlabel(r'Angle of Incidence $\theta_i$ (degrees)')
ax5.set_ylabel('Reflectance')
ax5.set_title(f'Fresnel Reflectances ($n$ = {n_exp:.2f})')
ax5.legend(fontsize=10)
ax5.set_xlim(0, 90)
ax5.set_ylim(0, 1)

fig5.tight_layout()
fig5.savefig('exercise3_fresnel.png', dpi=200, bbox_inches='tight')
plt.close(fig5)
print("Saved: exercise3_fresnel.png")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY OF RESULTS")
print("=" * 60)

print(f"\nExercise 1 (Two Polarizers):")
print(f"  Fit: I = I0 cos^2(theta - theta_0) + I_bg")
print(f"  I0      = {I0_fit:.3f} +/- {I0_err:.003f} V")
print(f"  theta_0 = {np.degrees(th0_fit):.2f} +/- {np.degrees(th0_err):.2f} deg")
print(f"  I_bg    = {Ibg_fit:.4f} +/- {Ibg_err:.4f} V")
print(f"  R^2     = {R2_1:.4f}")
print(f"  chi^2_r = {chi2r_1:.1f}  (dof = {dof1})")
print(f"  Linearized: slope = {popt_lin[0]:.3f} +/- {perr_lin[0]:.3f}")
print(f"              intercept = {popt_lin[1]:.4f} +/- {perr_lin[1]:.4f}")
print(f"              R^2 = {R2_lin:.4f}, chi^2_r = {chi2r_lin:.1f}")

print(f"\nExercise 2 (Three Polarizers):")
print(f"  Fit: I = A sin^2(2*phi) + I_bg")
print(f"  A (= I1/4) = {A_fit:.4f} +/- {A_err:.4f} V")
print(f"  I1 = 4A    = {I1_calc:.3f} +/- {I1_calc_err:.3f} V")
print(f"  phi_0      = {np.degrees(phi0_fit):.1f} +/- {np.degrees(phi0_err):.1f} deg")
print(f"  I_bg       = {Ibg2_fit:.5f} +/- {Ibg2_err:.5f} V")
print(f"  R^2        = {R2_2:.4f}")
print(f"  chi^2_r    = {chi2r_2:.2f}  (dof = {dof2})")

print(f"\nExercise 3 (Brewster's Angle):")
print(f"  theta_B = {theta_B_r} +/- {delta_thB_r} degrees")
print(f"  n       = {n_r} +/- {dn_r}")
print(f"  Accepted n_acrylic ~ 1.49")
print(f"  % diff  = {abs(n_exp - 1.49)/1.49 * 100:.1f}%")
