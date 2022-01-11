import pandas as pd
import timemachines
import numpy as np

TEMPLATE = 'https://raw.githubusercontent.com/microprediction/precisedata/main/returns/fathom_data_N.csv'
col = 'fathom_xx'

from timemachines.skaters.simple.movingaverage import EMA_SKATERS
from timemachines.skaters.simple.thinking import THINKING_SKATERS
from timemachines.skaters.simple.hypocraticensemble import HYPOCRATIC_ENSEMBLE_SKATERS
from timemachines.skating import prior
from timemachines.skatertools.utilities.conventions import targets

SKATERS = EMA_SKATERS + THINKING_SKATERS + HYPOCRATIC_ENSEMBLE_SKATERS
n_skaters = len(SKATERS)
n_burn = 100
print(n_skaters)

N=1

def residuals(f, y, k=1, e=100, n_burn=50):
    """ Feed fast skater all data points, then report residuals """
    assert n_burn>k
    es = [e for _ in y]
    x, _ = prior(f=f, y=y, k=k, e=es, x0=y[0])
    yt = targets(y)
    xk = [ xt[-1] for xt in x]
    return np.array(xk[n_burn:])-np.array(yt[n_burn:])

if __name__=='__main__':
    for N in range(350):
        try:
            df = pd.read_csv(TEMPLATE.replace('N',str(N)))
            y = df['fathom_xx'].values
            y = [ yt for yt in y if ~np.isnan(yt)]
            Z = None
            cols = [ f.__name__ for f in SKATERS ]
            df_out = pd.DataFrame(columns = cols )
            for f in SKATERS:
                z = residuals(f, y=y, k=1, e=100, n_burn = 400)
                df_out[f.__name__]=z
            name = 'skater_residuals_'+str(N)+'.csv'
            df_out.to_csv(name)
            print(name)
        except:
            pass




