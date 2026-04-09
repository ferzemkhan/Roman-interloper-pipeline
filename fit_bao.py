import os
import numpy as np
from pathlib import Path
from pycorr import TwoPointCorrelationFunction

from desilike import setup_logging
from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate, DampedBAOWigglesTracerCorrelationFunctionMultipoles
from desilike.observables.galaxy_clustering import TracerCorrelationFunctionMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.profilers import MinuitProfiler
from desilike.samplers import EmceeSampler
from desilike.samples import plotting

from Roman_config import *

def cut_matrix(cov, xcov, ellscov, xlim):
    assert len(cov) == len(xcov) * len(ellscov), f"Input matrix has size {len(cov)}, different than {len(xcov)} x {len(ellscov)}"
    indices = []
    ellscov = list(ellscov)
    for ell, limits in xlim.items():
        index = ellscov.index(ell) * len(xcov) + np.arange(len(xcov))
        index = index[(xcov >= limits[0]) & (xcov <= limits[1])]
        indices.append(index)
    indices = np.concatenate(indices, axis=0)
    return cov[np.ix_(indices, indices)]

def read_xi_cov(cov_txt):
    cov = np.genfromtxt(cov_txt)
    sedges = np.arange(cov_rmin, cov_rmax + cov_dr, cov_dr)
    smid = 0.5 * (sedges[1:] + sedges[:-1])
    slim = {ell: (smin, smax) for ell in ells}
    return cut_matrix(cov, smid, (0, 2, 4), slim)

def get_desilike_stats(xi_poles, xi_cov, fitting_method):
    template = BAOPowerSpectrumTemplate(z=zeff, fiducial='DESI', apmode=apmode)
    theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles(
        template=template,
        mode=recon_mode,
        smoothing_radius=smoothing_radius,
        broadband=broadband
    )
    observable = TracerCorrelationFunctionMultipolesObservable(
        data=xi_poles,
        covariance=xi_cov,
        theory=theory,
        ells=ells,
        wmatrix={'resolution': 1},
        slim={ell: (smin, smax, ds) for ell in ells}
    )
    likelihood = ObservablesGaussianLikelihood(observables=[observable])

    for param in likelihood.all_params.select(basename=['al*_*', 'bl*_*']):
        if fitting_method == 'profiling':
            param.update(derived='.auto')
        elif fitting_method == 'sampling':
            param.update(derived='.prec')

    if free_damping:
        for param in likelihood.all_params.select(basename='sigma*'):
            param.update(fixed=False)
    else:
        likelihood.all_params['sigmas'].update(value=sigmas, fixed=True)
        likelihood.all_params['sigmapar'].update(value=sigmapar, fixed=True)
        likelihood.all_params['sigmaper'].update(value=sigmaper, fixed=True)

    print(likelihood.varied_params.names())
    return theory, observable, likelihood

def profile(observable, likelihood):
    profiler = MinuitProfiler(likelihood, seed=1234)
    profiles = profiler.maximize(niterations=10)
    params = likelihood.varied_params.select(basename=['qiso', 'qap', 'qpar', 'qper',])
    if params: profiles = profiler.interval(params=params)
    likelihood(**profiles.bestfit.choice(input=True))
    if profiler.mpicomm.rank == 0:
        #plt.style.use(['enrique-science', 'retro'])
        state = observable.__getstate__()
        for name in ['data', 'theory', 'std', 'covariance', 'theory_nobao']:
                state[name] = getattr(observable, name)
        profiles.attrs['observable'] = state
        profiles.save(Path(mcmc_dir) / f'profiles-{tracer}_{fn_flags}.npy')
        profiles.to_stats(tablefmt='pretty', fn=Path(mcmc_dir) / f'profiles-{tracer}_{fn_flags}_stats.txt')
        fig = observable.plot()
        ax = fig.axes
        for aa in ax:
            aa.grid(visible=False)
            aa.xaxis.label.set_size(20)
            aa.yaxis.label.set_size(20)
        ax[0].set_title(f'Roman-{tracer}'rf'${zmin} < z < {zmax}$')
        fig.savefig(Path(mcmc_dir) / f'profiles-{tracer}_{fn_flags}_multipoles.png', bbox_inches='tight', dpi=300)
        fig = observable.plot_bao()
        ax = fig.axes
        for aa in ax:
            aa.grid(visible=False)
            aa.xaxis.label.set_size(15)
            aa.yaxis.label.set_size(15)
        ax[0].set_title(f'Roman-{tracer}'rf'${zmin} < z < {zmax}$')
        fig.savefig(Path(mcmc_dir) / f'profiles-{tracer}_{fn_flags}_BAO.png', bbox_inches='tight', dpi=300)

def sample(likelihood):
    outdir = Path(mcmc_dir)
    output_fn = [outdir / f"Roman-{tracer}_BAO_chain_z{zmin}-{zmax}_{i}.npy" for i in range(nchains)]
    sampler = EmceeSampler(likelihood, nwalkers=40, save_fn=output_fn, seed=1234)
    chains = sampler.run(min_iterations=200, max_iterations=10000, check={'max_eigen_gr': 0.01})
    if sampler.mpicomm.rank == 0:
        chain = chains[0].concatenate([c.remove_burnin(0.5)[::10] for c in chains])
        if apmode == 'qparqper':
            qiso = (chain['qpar']**(1./3.) * chain['qper']**(2./3.)).clone(param=dict(basename='qiso', derived=True, latex=r'q_{\rm iso}'))
            qap = (chain['qpar'] / chain['qper']).clone(param=dict(basename='qap', derived=True, latex=r'q_{\rm AP}'))
            chain.set(qiso)
            chain.set(qap)
        if apmode == 'qisoqap':
            qpar = (chain['qiso'] * chain['qap']**(2/3)).clone(param=dict(basename='qpar', derived=True, latex=r'q_{\parallel}'))
            qper = (chain['qiso'] * chain['qap']**(-1/3)).clone(param=dict(basename='qper', derived=True, latex=r'q_{\perp}'))
            chain.set(qpar)
            chain.set(qper)

        stats_fn = outdir / f"Roman-{tracer}_BAO_chain_z{zmin}-{zmax}_stats.txt"
        tri_fn   = outdir / f"Roman-{tracer}_BAO_chain_z{zmin}-{zmax}_triangle.png"

        chain.to_stats(tablefmt='pretty', fn=stats_fn)
        plot_params = [p for p in ['qpar', 'qper', 'qiso', 'qap'] if p in chain]
        markers = {p: 1.0 for p in plot_params}
        plotting.plot_triangle(chain, fn=tri_fn, title_limit=1, params=plot_params, markers=markers)

def main(fitting_method=fitting_method):
    ensure_dirs()
    setup_logging()

    xi_poles = TwoPointCorrelationFunction.load(rascalc_2pcf_name)
    xi_cov = read_xi_cov(rescaled_cov_txt)

    print("xi_poles shape:", xi_poles.shape)
    print("Covariance shape:", xi_cov.shape)

    theory, observable, likelihood = get_desilike_stats(xi_poles, xi_cov, fitting_method)

    print("Covariance shape:", xi_cov.shape)
    print("Observable flat data length:", observable.flatdata.size)

    if fitting_method == "profiling":
        profile(observable, likelihood)
    elif fitting_method == "sampling":
        sample(likelihood)
    else:
        raise ValueError(f"Unknown fitting_method = {fitting_method}")

if __name__ == "__main__":
    main()