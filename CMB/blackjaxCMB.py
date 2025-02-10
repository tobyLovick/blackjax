# | # Install instructions
# |
# |```bash
# |python -m venv venv
# |source venv/bin/activate
# |pip install tqdm numpy jax anesthetic
# |pip install git+https://github.com/handley-lab/blackjax@nested_sampling
# |python blackjaxCMB.py
# |```
#

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import blackjax
import blackjax.ns.adaptive

#| You may need to set jax to double, due to the flatness of the likelihood function near the peak
jax.config.update('jax_enable_x64', True) 

paramnames = [r'$\Omega_b h^2$',r'$\Omega_c h^2$',r'$h$',r'$\tau$',r'$n_s$',r'$\ln(10^{10}A_s)$']
params = ['Ωbh2', 'Ωch2', 'h', 'τ', 'ns', 'lnA']
#| Seed set for reproducibility
rng_key = jax.random.PRNGKey(0)

#| Define the loglikelihood and logprior functions
d = 6 # Dimension of the problem
#| loglikelihood function
class CMB(object):
    def __init__(self, Cl):
        self.Cl = Cl
        self.l = jnp.arange(2, 2509)

    def rvs(self, shape=()):
        shape = tuple(jnp.atleast_1d(shape))
        return jax.random.chisquare(jax.random.PRNGKey(0), 2*self.l+1, shape + self.Cl.shape) * self.Cl / (2*self.l+1)
    def logpdf(self, x):
        return (jax.scipy.stats.chi2.logpdf((2*self.l+1)*x/self.Cl, 2*self.l+1) + jnp.log(2*self.l+1) - jnp.log(self.Cl)).sum(axis=-1)

from cosmopower_jax.cosmopower_jax import CosmoPowerJAX 
emulator = CosmoPowerJAX(probe='cmb_tt')
planckparams = jnp.array([0.02225, 0.1198, 0.693, 0.097, 0.965, 3.05])
d_obs = CMB(emulator.predict(planckparams)).rvs()

## save d_obs to file for comparison with other analyses
np.save('d_obs.npy', d_obs)

def loglikelihood_fn(x):
    return CMB(jnp.array(emulator.predict(x))).logpdf(d_obs)

parammin, parammax = jnp.array([[0.01865, 0.02625], [0.05, 0.255], [0.64, 0.82], [0.04, 0.12], [0.84, 1.1], [1.61, 3.91]]).T ## Prior set by the emulator's training range (cosmo_power_jax) 

def logprior_fn(x): ## 6D Uniform prior
    return jax.scipy.stats.uniform.logpdf(x, parammin, parammax).sum()

#| Define the Nested Sampling algorithm
n_live = 500
n_delete = 20
num_mcmc_steps = d * 5
ndead_max = 1000

#| Initialize the Nested Sampling algorithm
algo = blackjax.ns.adaptive.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    n_delete=n_delete,
    num_mcmc_steps=num_mcmc_steps,
)

@jax.jit
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = algo.step(subk, state)
    return (state, k), dead_point

#| Sample live points from the prior
rng_key, init_key = jax.random.split(rng_key, 2)
initial_particles= jax.random.uniform(init_key, (n_live, d)) * (parammax - parammin) + parammin
state = algo.init(initial_particles, loglikelihood_fn)

#| Run the Nested Sampling algorithm
import tqdm
dead = []
n_dead = 0
pbar = tqdm.tqdm(desc="Dead Points", unit="dead points")  # No total specified
while not state.sampler_state.logZ_live - state.sampler_state.logZ < -5: # Alternate Termination Criteria are possible
    (state, rng_key), dead_info = one_step((state, rng_key), None)
    dead.append(dead_info)
    pbar.update(n_delete)  # Update progress bar
pbar.close()
#| anesthetic post-processing
from anesthetic import NestedSamples
import numpy as np
dead = jax.tree.map(lambda *args: jnp.concatenate(args), *dead)
live = state.sampler_state
logL = np.concatenate((dead.logL, live.logL), dtype=float)
logL_birth = np.concatenate((dead.logL_birth, live.logL_birth), dtype=float)
data = np.concatenate((dead.particles, live.particles), dtype=float)
samples = NestedSamples(data, logL=logL, logL_birth=logL_birth, columns=params, labels=paramnames)
samples.to_csv('jaxLCDM.csv')

#| Basic Plotting Code
from anesthetic import make_2d_axes
params = ['Ωbh2', 'Ωch2', 'h', 'τ', 'ns', 'lnA']
fig, axes = make_2d_axes(params, upper = False)

samples.plot_2d(axes,c = '#CC3311', #Plotting with recommended resolution settings
                      kind={'lower': 'kde_2d', 'diagonal': 'kde_1d'},
                      diagonal_kwargs={'nplot_1d': 1000},
                      lower_kwargs={'nplot_2d': 100**2},
                      ncompress='entropy',label='JAX samples')

for i in range(6): # add truth lines at the input parameters used to generate the data
    for j in range(i+1):
        ax = axes.loc[params[i], params[j]]
        i!=j and ax.axhline(planckparams[i], color='k', linestyle='--')
        ax.axvline(planckparams[j], color='k', linestyle='--',label='Ground Truth')

axes.iloc[-1, 0].legend(loc='lower center', bbox_to_anchor=(len(axes)/2, len(axes)), ncol=6)
fig.savefig('jaxLCDM.pdf', format="pdf", bbox_inches = 'tight') 