import numpy as n
import scipy.io as sio
import matplotlib.pyplot as plt
import glob
import bz2
import stuffr
import os

#fl=glob.glob("/urdr/data/test/20211213/ipy_fixed42m_4.1l_NO@32p/*/*.bz2")

fl=glob.glob("/urdr/data/test/ipy_fixed42m_4.1l_CP@32p/20220330_*/*.bz2")

# 
#fl=glob.glob("/urdr/data/test/ipy_fixed42m_4.1l_NO@32p/*/*.bz2")
fl.sort()

odir="/urdr/scratch/ipypl/20220330"
os.system("mkdir -p %s"%(odir))

print(fl)

#start_time=stuffr.date2unix(2022,3,2,19,0,0)
start_time=0#stuffr.date2unix(2022,3,2,19,0,0)


bf=bz2.open(fl[0],"r")
d=sio.loadmat(bf)
d_data=n.copy(d["d_data"][:,:])
d_data[:,:]=0.0
bf.close()


pl_len=86627

n_avg=10
n_tot=0
for f in fl:
    print(f)
    bf=bz2.open(f,"r")
    d=sio.loadmat(bf)
    center_frequencies=d["d_parbl"][0,46:50]
    pl_len=86627

    print(center_frequencies)
    print(d.keys())
#    offset=20
 #   n_lags=1153
  #  n_range_gate = 66
    t0=d["d_parbl"][0,10]
    

    

    # these are 0.4 microsecond lags between 0 and 29.6 microseconds
    # why? nobody really knows
    n_short_lags = 75
    n_short_lag_range_gates = 67
    
    # more lags, now between 0.4 and 1152*0.4 microseconds
    # why? nobody really knows
    n_long_lags = 1152
    n_long_lag_range_gates = 66
    

    F=n.fft.fft(n.zeros(n_long_lags,dtype=n.complex64))
    n_fft=len(F)
    P=n.zeros([n_long_lag_range_gates,n_fft])

    if t0 >= start_time:

        d_data = d_data + d["d_data"][:,:]
        n_tot+=1
        
    print(n_tot)
    print(n_avg)
    if n_tot == n_avg:
        n_tot=0
        
        for ci in range(4):
            offset=20
            z_short_lags=d_data[(offset + 86627*ci):(offset + 86627*ci + n_short_lags*n_short_lag_range_gates) ,0]
            z_short_lags.shape=(n_short_lags,n_short_lag_range_gates)
 #           z_short_lags.shape=(n_short_lag_range_gates,n_short_lags)         

            # 
            offset=20 + n_short_lags*n_short_lag_range_gates       
            z_long_lags=d_data[(offset + 86627*ci):(offset + 86627*ci + n_long_lags*n_long_lag_range_gates) ,0]
            z_long_lags.shape=(n_long_lags,n_long_lag_range_gates)
            
            for ri in range(n_long_lag_range_gates):
                z_long_lags[:,ri]=n.linspace(1,0,num=n_long_lags)*z_long_lags[:,ri]
                
            #plt.pcolormesh(n.transpose(n.real(z_long_lags)))
            #plt.colorbar()
            #plt.show()
            
            
            
            
#            z.shape=(n_lags,n_range_gate)
            z=n.transpose(z_long_lags)
            for ri in range(n_long_lag_range_gates):
                P[ri,:]=n.abs(n.fft.fftshift(n.fft.fft(z[ri,:])))

            dB=10.0*n.log10(P)
            nfloor=n.nanmedian(dB)
            plt.pcolormesh(dB,vmin=nfloor-3,vmax=nfloor+10)
            plt.colorbar()
            plt.title("pl channel %1.2f (MHz)\n%s"%(center_frequencies[ci],stuffr.unix2datestr(t0)))
            plt.tight_layout()
#            plt.show()
            plt.savefig("%s/pl-%02d-%1.2f.png"%(odir,ci,t0))
            plt.clf()
            plt.close()
        # done processing this integration period
        d_data[:,:]=0.0
        
    bf.close()

