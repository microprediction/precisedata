import pandas as pd
from precise.universe import sp_tickers, sp_table
from runthis import parse_kwargs
import random
import quantstats as qs
import numpy as np

TABLE = sp_table()
TICKERS = TABLE['Symbol'].values
RETURNS = dict()


def make_long(chunk=50):

    for i1,ticker1 in enumerate(TICKERS):
        corrs = list()
        covs = list()
        var0s = list()
        var1s = list()
        p0s = list()
        p1s = list()
        p01s = list()
        w0s = list()
        w1s = list()
        if ticker1 not in RETURNS:
            RETURNS[ticker1] = qs.utils.download_returns(ticker=ticker1, period='max').values[1:]
        for i2,ticker2 in enumerate(TICKERS):
            if ticker1 != ticker2:
                print(ticker1+'-'+ticker2)
                if ticker2 not in RETURNS:
                    import time
                    time.sleep(1)
                    RETURNS[ticker2] = qs.utils.download_returns(ticker=ticker2, period='max').values[1:]
                r1 = RETURNS[ticker1]
                r2 = RETURNS[ticker2]
                n1 = len(r1)
                n2 = len(r2)
                n = min(n1,n2)
                r1 = r1[:n]
                r2 = r2[:n]
                if n>10000:
                    step = chunk
                    for k in range(0,n,step):
                        r1_chunk = r1[k:k+step]
                        r2_chunk = r2[k:k+step]
                        try:
                          the_cov = np.cov(np.array([r1_chunk,r2_chunk]))
                        except ValueError:
                            print({'n1':n1,'n2':n2})
                        the_corr = np.corrcoef( np.array([r1_chunk,r2_chunk]))
                        try:
                            the_precision = np.linalg.inv(the_cov)
                            okay = True
                        except np.linalg.LinAlgError:
                            okay = False

                        if okay:

                            the_weights = np.squeeze(np.matmul(the_precision,np.ones(shape=(2,1))))
                            the_weights = the_weights/sum(the_weights)

                            corrs.append(the_corr[0,1])
                            covs.append(10000*the_cov[0,1])
                            var0s.append(10000*the_cov[0,0])
                            var1s.append(10000*the_cov[1, 1])
                            p0s.append(the_precision[0,0]/1000.)
                            p1s.append(the_precision[1,1]/1000.)
                            p01s.append(the_precision[0,1]/1000.)
                            w0s.append(the_weights[0])
                            w1s.append(the_weights[1])

                            if random.choice(list(range(1000)))==1:
                                print(the_precision[0,1]/1000.)


        df = pd.DataFrame(columns=['fathom_rho','fathom_xx','fathom_xy','fathom_yy','fathom_00','fathom_01','fathom_11','fathom_w'])
        df['fathom_rho'] = corrs
        df['fathom_xx'] = var0s
        df['fathom_xy'] = covs
        df['fathom_yy'] = var1s
        df['fathom_00'] = p0s
        df['fathom_11'] = p1s
        df['fathom_01'] = p01s
        df['fathom_w'] = w0s
        print('Saving ...')
        df.to_csv('fathom_data_'+str(i1)+'.csv')


if __name__=='__main__':
    make_long(chunk=50)




