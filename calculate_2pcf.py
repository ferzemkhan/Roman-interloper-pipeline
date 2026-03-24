import numpy as np
import pandas as pd
from pycorr import TwoPointCorrelationFunction, KMeansSubsampler
from cosmoprimo.fiducial import DESI
from build_data_and_random import build_sample, build_randoms

from Roman_config import *

def compute_2pcf():
    ensure_dirs()

    cosmo = DESI()
    data = build_sample(tracer)
    rand = build_randoms(data)

    print("Tracer:", tracer)
    print("Data shape:", data.shape)
    print("Random shape:", rand.shape)
    print("Jackknife regions:", njack)
    print("NTHREADS:", NTHREADS)

    subsampler = KMeansSubsampler(
        mode='angular',
        positions=[rand[0], rand[1], cosmo.comoving_radial_distance(rand[2])],
        nsamples=njack,
        random_state=jack_seed,
        position_type='rdd'
    )
    galaxy_jack = subsampler.label(
        [data[:, 0], data[:, 1], cosmo.comoving_radial_distance(data[:, 2])],
        position_type='rdd'
    )
    random_jack = subsampler.label(
        [rand[0], rand[1], cosmo.comoving_radial_distance(rand[2])],
        position_type='rdd'
    )

    np.save(data_name, data)
    np.save(rand_name, rand)
    np.save(galaxy_jack_name, galaxy_jack)
    np.save(random_jack_name, random_jack)
    
    print("Saved data positions:", data_name)
    print("Saved random positions:", rand_name)
    print("Saved galaxy jackknife labels:", galaxy_jack_name)
    print("Saved random jackknife labels:", random_jack_name)

    s_edges = np.linspace(s_edges_min, s_edges_max, s_edges_nbin + 1)
    mu_edges = np.linspace(-1.0, 1.0, mu_edges_nbin + 1)

    tmp = TwoPointCorrelationFunction(
        mode='smu',
        edges=(s_edges, mu_edges),
        data_positions1=(data[:, 0], data[:, 1], cosmo.comoving_radial_distance(data[:, 2])),
        randoms_positions1=(rand[0], rand[1], cosmo.comoving_radial_distance(rand[2])),
        data_samples1=galaxy_jack,
        randoms_samples1=random_jack,
        position_type="rdd",
        engine="corrfunc",
        D1D2=None,
        gpu=False,
        nthreads=NTHREADS
    )

    if mbin_cf:
        assert tmp.shape[1] % (2 * mbin_cf) == 0, "Angular rebinning is not possible"
    xi_table_11 = tmp[
        ::max(1, tmp.shape[0] // nbin_cf),
        ::(tmp.shape[1] // 2 // mbin_cf if mbin_cf else 1)
    ].wrap()

    if mbin:
        assert tmp.shape[1] % (2 * mbin) == 0, "Angular rebinning is not possible"
    test = tmp[
        ::tmp.shape[0] // nbin,
        ::(tmp.shape[1] // 2 // mbin if mbin else 1)
    ].wrap()

    tmp.save(raw_2pcf_name)
    xi_table_11.save(xi_table_name)
    test.save(rascalc_2pcf_name)

    print("Saved raw pycorr object:", raw_2pcf_name)
    print("Saved xi_table_11 for RascalC:", xi_table_name)
    print("Saved pycorr_allcounts_11 object:", rascalc_2pcf_name)

    print("tmp shape:", tmp.shape)
    print("xi_table_11 shape:", xi_table_11.shape)
    print("test shape:", test.shape)


def main():
    compute_2pcf()

if __name__ == "__main__":
    main()