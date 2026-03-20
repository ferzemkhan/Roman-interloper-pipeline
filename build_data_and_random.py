import numpy as np
import pandas as pd

from roman_config import *

def build_sample(tracer_name):
    df = pd.read_csv(catalog_path)

    ra_d = df["ra"].to_numpy()
    dec_d = df["dec"].to_numpy()
    z_d = df["z_obs"].to_numpy()

    mask = (z_d >= zcat_min) & (z_d < zcat_max)
    ra_d, dec_d, z_d = ra_d[mask], dec_d[mask], z_d[mask]

    z_s3 = (1. + z_d) * L_ratio - 1.
    rng = np.random.default_rng(catalog_seed)
    rand_num = rng.uniform(size=len(z_d))

    percent_idx = np.where(rand_num < interloper_percent)
    z_obs1 = z_d.copy()
    z_obs1[percent_idx] = z_s3[percent_idx]

    index_c = np.where((z_d >= zmin) & (z_d < zmax))[0]
    index_obs1 = np.where((z_obs1 >= zmin) & (z_obs1 < zmax))[0]
    index_93c = np.intersect1d(index_c, index_obs1)

    data_c = np.c_[ra_d[index_c], dec_d[index_c], z_d[index_c]]
    data_93c = np.c_[ra_d[index_93c], dec_d[index_93c], z_d[index_93c]]
    data_obs1 = np.c_[ra_d[index_obs1], dec_d[index_obs1], z_obs1[index_obs1]]

    if tracer_name == "fullcorrect":
        return data_c
    elif tracer_name == "93correct":
        return data_93c
    elif tracer_name == "observe1":
        return data_obs1
    else:
        raise ValueError(f"Unknown tracer = {tracer_name}")

def build_randoms(data):
    df_r = pd.read_csv(random_catalog_path)
    ra_r = df_r["ra"].to_numpy()
    dec_r = df_r["dec"].to_numpy()

    nr_target = random_factor * len(data)
    rng = np.random.default_rng(catalog_seed)
    idx = rng.choice(len(ra_r), size=nr_target, replace=False)

    rand = np.vstack((
        ra_r[idx],
        dec_r[idx],
        rng.choice(data[:, 2], size=len(idx), replace=True)
    ))
    return rand

def main():
    data = build_sample(tracer)
    rand = build_randoms(data)

    np.save(data_name, data)
    np.save(rand_name, rand)

    print(f"Saved data to {data_name}")
    print(f"Saved randoms to {rand_name}")


if __name__ == "__main__":
    main()