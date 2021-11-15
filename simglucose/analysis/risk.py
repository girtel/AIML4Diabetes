import numpy as np
import warnings


def risk_index(BG, horizon):
    # BG is in mg/dL
    # horizon in samples
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        BG_to_compute = BG[-horizon:]
        #print('BG=',BG,'BG_to_compute=',BG_to_compute)
        fBG = 1.509 * (np.log(BG_to_compute)**1.084 - 5.381)
        rl = 10 * fBG[fBG < 0]**2
        rh = 10 * fBG[fBG > 0]**2
        LBGI = np.nan_to_num(np.mean(rl))
        HBGI = np.nan_to_num(np.mean(rh))
        RI = LBGI + HBGI
        #print('LBGI=',LBGI,'HBGI=',HBGI,'RI=',RI)
    return (LBGI, HBGI, RI)

def magni_RI(BG, horizon):
  BG_to_compute = BG[-horizon:]
  c0 = 3.35506
  c1 = 0.8353
  c2 = 3.7932
  logbPowerc1 = np.power(np.log(BG_to_compute),c1)
  RI = 10*(np.power(c0*(logbPowerc1-c2),2))
  return RI
