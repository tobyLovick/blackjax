from anesthetic import read_chains,make_2d_axes

samples = read_chains('jaxLCDM.csv')
params = ['Ωbh2', 'Ωch2', 'h', 'τ', 'ns', 'lnA']
fig, axes = make_2d_axes(params, upper = False)

samples.plot_2d(axes,c = '#CC3311', #Plotting with recommended resolution settings
                      kind={'lower': 'kde_2d', 'diagonal': 'kde_1d'},
                      diagonal_kwargs={'nplot_1d': 1000},
                      lower_kwargs={'nplot_2d': 100**2},
                      ncompress='entropy',label='JAX samples')

## add truth lines at the input parameters used to generate the data
planckparams = [0.02225, 0.1198, 0.693, 0.097, 0.965, 3.05]
for i in range(6):
    for j in range(i+1):
        ax = axes.loc[params[i], params[j]]
        i!=j and ax.axhline(planckparams[i], color='k', linestyle='--')
        ax.axvline(planckparams[j], color='k', linestyle='--',label='Ground Truth')

axes.iloc[-1, 0].legend(loc='lower center', bbox_to_anchor=(len(axes)/2, len(axes)), ncol=6)

fig.savefig('jaxLCDM.pdf', format="pdf", bbox_inches = 'tight') 