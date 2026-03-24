import os
from pathlib import Path

# =========================
# Main science choices
# =========================
tracer = "fullcorrect"
compare_tracers = ["fullcorrect", "observe1"]
interloper_percent = 0.07
periodic_boxsize = 0

zmin, zmax = 1.5, 1.8
zeff = 0.5 * (zmin + zmax)

# 2PCF / covariance binning
s_edges_min = 0.0
s_edges_max = 200.0
s_edges_nbin = 40                  # 40 radial bins -> ds = 5
mu_edges_nbin = 200                # full mu resolution in pycorr

nbin_cf = 40
mbin_cf = None                     # None = no mu smoothing in xi_table_11
nbin = 40
mbin = None                        # None = no mu smoothing in pycorr_allcounts_11
skip_nbin_pre = 0

# Jackknife / RascalC
jackknife = True
njack = 60
NTHREADS = 128

mode = "legendre_projected"
max_l = 4
N2 = 5
N3 = 10
N4 = 20

# IMPORTANT: needs multiple samples
n_loops = 512
loops_per_sample = 128

# # Early RascalC cut: keep 50–200
# skip_s_bins = (10, 0)

# # Final exported covariance starts at 60 Mpc/h
# skip_r_bins = 12
# skip_l = 0

# BAO fit range
cov_rmin = 60.0
cov_rmax = 200.0
cov_dr = (s_edges_max - s_edges_min) / s_edges_nbin
# cov_dr = 5.0

skip_s_bins = (
    int((cov_rmin - s_edges_min) / cov_dr),
    int((s_edges_max - cov_rmax) / cov_dr)
)

r_step = int(cov_dr)
rmin_real = int(cov_dr * skip_s_bins[0])
xilabel = "".join(str(i) for i in range(0, max_l + 1, 2))

smin, smax, ds = 60.0, 150.0, 5.0
ells = (0, 2)

apmode = "qisoqap"
recon_mode = ""
smoothing_radius = 15
broadband = "power"

sigmas = 3.0
sigmapar = 10.0
sigmaper = 6.0
free_damping = True

nchains = 8
# fitting_method = "profiling"   # or "sampling"
fitting_method = "sampling"

# Catalog choices
catalog_seed = 1234
jack_seed = 1234
rascalC_seed = 1234
random_factor = 6

zcat_min, zcat_max = 0.0, 1.9

L_Ha = 6549.
L_S3 = 9531.
L_ratio = L_S3 / L_Ha

# =========================
# Paths
# =========================
base_dir = Path("/pscratch/sd/y/yuyuwang/Roman_data")
bao_fit_dir = base_dir / "BAO_fit"
two_pt_dir = base_dir / "2PT_result"

catalog_path = base_dir / "sfr_l2_catalog_all_redshiftspace.csv"
random_catalog_path = base_dir / "diffsky_LastJourney_random_catalog_all.csv"

cov_base = f"Roman_interloper_BAOfit_test-{tracer}_z{zmin}-{zmax}_fixed"
cov_dir = bao_fit_dir / cov_base
cov_tmp_dir = cov_dir / "tmp"
cov_txt_dir = cov_dir / "cov_txt"
mcmc_dir = bao_fit_dir / "MCMC_chain"

data_name = two_pt_dir / f"data_positions_{tracer}_z{zmin}-{zmax}.npy"
rand_name = two_pt_dir / f"random_positions_{tracer}_z{zmin}-{zmax}.npy"
galaxy_jack_name = two_pt_dir / f"galaxy_jack_{tracer}_z{zmin}-{zmax}_j{njack}.npy"
random_jack_name = two_pt_dir / f"random_jack_{tracer}_z{zmin}-{zmax}_j{njack}.npy"

# Saved pycorr objects
raw_2pcf_name = two_pt_dir / f"allcounts_raw_{tracer}_z{zmin}-{zmax}.npy"
xi_table_name = two_pt_dir / f"xi_table_11_{tracer}_z{zmin}-{zmax}.npy"
rascalc_2pcf_name = two_pt_dir / f"2PCF_forRascalC_z{zmin}-{zmax}_{tracer}_njack-{njack}.npy"

# RascalC processed files
raw_cov_name = cov_dir / f"Raw_Covariance_Matrices_n{nbin}_l{max_l}.npz"
gaussian_npz = cov_dir / f"Rescaled_Covariance_Matrices_Legendre_n{nbin}_l{max_l}.npz"
jackknife_npz = cov_dir / f"Rescaled_Covariance_Matrices_Legendre_Jackknife_n{nbin}_l{max_l}_j{njack}.npz"

gaussian_cov_txt = cov_txt_dir / f"xi{xilabel}_{tracer}_{zmin}_{zmax}_lin{r_step}_s{rmin_real}-{int(cov_rmax)}_cov_RascalC_Gaussian.txt"
rescaled_cov_txt = cov_txt_dir / f"xi{xilabel}_{tracer}_{zmin}_{zmax}_lin{r_step}_s{rmin_real}-{int(cov_rmax)}_cov_RascalC_rescaled.txt"

# BAO filenames
fn_flags = f"Roman-{tracer}_z{zmin}-{zmax}_{apmode}_s{smin}-{smax}_ds{int(ds)}_ells{''.join(map(str, ells))}"

def ensure_dirs():
    cov_dir.mkdir(parents=True, exist_ok=True)
    cov_tmp_dir.mkdir(parents=True, exist_ok=True)
    cov_txt_dir.mkdir(parents=True, exist_ok=True)
    two_pt_dir.mkdir(parents=True, exist_ok=True)
    mcmc_dir.mkdir(parents=True, exist_ok=True)