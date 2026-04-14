import numpy as np
from pycorr import TwoPointCorrelationFunction
from cosmoprimo.fiducial import DESI
from RascalC import run_cov
from RascalC.cov_utils import export_cov_legendre
from RascalC.post_process.legendre import post_process_legendre
from RascalC.post_process.legendre_mix_jackknife import post_process_legendre_mix_jackknife
from RascalC.convergence_check_extra import convergence_check_extra

from Roman_config import *

def run_rascalc():
    ensure_dirs()

    # Load saved pycorr objects from calculate_2pcf.py
    xi_table_11 = TwoPointCorrelationFunction.load(xi_table_name)
    pycorr_allcounts_11 = TwoPointCorrelationFunction.load(rascalc_2pcf_name)

    # Load saved catalog / jackknife arrays from calculate_2pcf.py
    rand = np.load(rand_name)
    random_jack = np.load(random_jack_name)

    print("Loaded xi_table_11:", xi_table_name)
    print("Loaded pycorr_allcounts_11:", rascalc_2pcf_name)
    print("Loaded random positions:", rand_name)
    print("Loaded random jackknife labels:", random_jack_name)

    print("xi_table_11 shape:", xi_table_11.shape)
    print("pycorr_allcounts_11 shape:", pycorr_allcounts_11.shape)
    print("rand shape:", rand.shape)
    print("random_jack shape:", random_jack.shape)
    print("NTHREADS:", NTHREADS)

    cosmo = DESI()

    randoms_positions1 = np.vstack([
        rand[0],
        rand[1],
        cosmo.comoving_radial_distance(rand[2])
    ])
    randoms_weights1 = np.ones(randoms_positions1.shape[1], dtype="f8")
    randoms_samples1 = random_jack.astype(int)

    print("randoms_positions1 shape:", randoms_positions1.shape)
    print("randoms_weights1 shape:", randoms_weights1.shape)
    print("randoms_samples1 shape:", randoms_samples1.shape)

    results = run_cov(
        mode=mode,
        max_l=max_l,
        nthread=NTHREADS,
        boxsize=periodic_boxsize,
        N2=N2,
        N3=N3,
        N4=N4,
        n_loops=n_loops,
        loops_per_sample=loops_per_sample,
        out_dir=str(cov_dir),
        tmp_dir=str(cov_tmp_dir),
        pycorr_allcounts_11=pycorr_allcounts_11,
        xi_table_11=xi_table_11,
        randoms_positions1=randoms_positions1,
        randoms_weights1=randoms_weights1,
        randoms_samples1=randoms_samples1,
        position_type="rdd",
        skip_s_bins = skip_s_bins,
        normalize_wcounts = True
        # seed=rascalC_seed
    )
    print("Cutting range:", skip_s_bins)

    cov = results["full_theory_covariance"]
    print("Covariance shape from run_cov:", cov.shape)
    print("Shot-noise rescaling:", results.get("shot_noise_rescaling", None))
    print("RascalC output directory:", cov_dir)
    print("RascalC tmp directory:", cov_tmp_dir)

    xi_jack_name = os.path.join(cov_dir, "xi_jack", f"xi_jack_n{nbin}_m*_j{njack}_11.dat")
    jack_results = post_process_legendre_mix_jackknife(
        str(xi_jack_name),
        str(cov_dir / "weights"),
        str(cov_dir),
        mbin,
        max_l,
        str(cov_dir),
        skip_r_bins=skip_s_bins,
        skip_l=0,
        print_function=True
    )

    export_cov_legendre(str(jackknife_npz), max_l, str(rescaled_cov_txt))
    print("Saved jackknife/rescaled txt:", str(rescaled_cov_txt))
    convergence_check_extra(jack_results, print_function = True)

    gaussian_results = post_process_legendre(
        str(cov_dir),
        nbin,
        max_l,
        str(cov_dir),
        skip_r_bins=skip_s_bins,
        print_function=print
    )
    print("Gaussian full_theory_covariance shape:", gaussian_results["full_theory_covariance"].shape)
    print("Saved Gaussian npz:", str(gaussian_npz))
    
    export_cov_legendre(str(gaussian_npz), max_l, str(gaussian_cov_txt))
    print("Saved Gaussian txt:", str(gaussian_cov_txt))
    convergence_check_extra(gaussian_results, print_function = True)

def main():
    run_rascalc()

if __name__ == "__main__":
    main()