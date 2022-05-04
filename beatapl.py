#!/usr/bin/env python
#
# Simple plasma-line analysis 
#
import numpy as n
import scipy.io as sio
import matplotlib.pyplot as plt
import stuffr
import bz2
import os
import glob

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#dname="/data2/see_pl/beata_cp1_2.0u_NO@uhf"
#dname="/data2/march2016/beata_cp1_2.0u_SP@uhf"
#dname="/data2/nov2014/beata_cp1_2.0u_NO@uhf"
#dname="/data2/march2016/beata_cp1_2.0u_SP@uhf"
#dname="/data2/upflow_20190905/beata_cp1_2.0u_AA@uhf"

# directory where *.mat.bz2 eiscat data files are located
#dname="/data2/march_10.03.2016/beata_cp1_2.0u_SP@uhf"

dname="/data0/2014.02.20/beata_cp1_2.0u_SP@uhf"


# number of files to average (beata is 5 seconds per file)
n_avg=2
# directory where files and plots are written
out_prefix="10s"

# make a list of all files
fl=glob.glob("%s/*/*.mat.bz2"%(dname))
fl.sort()
print(fl)

def calc_pl(fl,out_prefix="5s",smooth_freq=2.0):
    """
    fl = list of files to analyze
    smooth_freqs = how much do you smooth in frequency
    """
    os.system("mkdir -p %s/%s"%(dname,out_prefix))
    a=sio.loadmat(bz2.BZ2File(fl[0],"r"))
    d=a["d_data"][:,0]
    t0=a["d_parbl"][0,10]
    t1=a["d_parbl"][0,11]
    print(len(d))
    if len(fl)>1:
        d[:]=0.0
        for f in fl:
            a=sio.loadmat(bz2.BZ2File(f,"r"))
            d+=a["d_data"][:,0]

    # window function. scale by the squareroot of the number of samples.
    # to make acf estimation noise white (each lag has the same noise variance)
    wf=n.sqrt(n.arange(640,320,-0.4))

    # determine hermitian fft length (assumes conjugate symmetric input)
    F=n.fft.hfft(n.repeat(1,800))
    n_fft=len(F)
    
    # go through all plasma-line channels
    for c in range(3):
        cf=a["d_parbl"][0,c+46]*1e6
        r0=107.25  # hard coded initial altitude
        # some kind of magic offset constant
        pl0=d[(36611+82232*c):(36611+82232*c+90*800)]
        pl0.shape=(800,90)
        pl0=n.transpose(pl0)
        P=n.zeros([90,n_fft])
        nr=90

        fvec=n.fft.fftshift(n.fft.fftfreq(n_fft,d=1.0/2.5e6))+cf
        rvec=n.arange(90)*3.0+r0

        # for each range gate, FFT the acf estimate to obtain spectrum estimate
        for ri in range(nr):
            P[ri,:]=n.convolve(n.repeat(1.0/smooth_freq,int(smooth_freq)),n.abs(n.fft.fftshift(n.fft.hfft(wf*pl0[ri,:]))),mode="same")

        P2=n.copy(P)
        noise_std=n.zeros(n_fft)
        for si in range(n_fft):
            P2[:,si]=P[:,si]/n.median(n.abs(P[:,si]))
            noise_std[si]=n.median(n.abs(P[:,si]))


        pl={"t0":t0,"datestr":stuffr.unix2datestr(t0),"dt":len(fl)*5.0,"range":rvec,"freqs":fvec,"noise_std":noise_std,"P":P,"P_norm":P2}

        # save result
        sio.savemat("%s/%s/pl_%d_%d.mat"%(dname,out_prefix,c,t0),pl)

        mean_est=n.median(P2)
        std_est=n.median(n.abs(P2-mean_est))
        plt.pcolormesh(fvec/1e6,rvec,P2,vmin=mean_est-1*std_est,vmax=mean_est+10*std_est,cmap="plasma")
        plt.title("%s"%(stuffr.unix2datestr(t0)))

        plt.xlabel("Frequency-offset (MHz)")
        plt.ylabel("Range (km)")
        plt.savefig("%s/%s/pl_%d_%d.png"%(dname,out_prefix,c,t0))        
        plt.close()
        plt.clf()

        
n_spec=len(fl)/n_avg

for i in range(comm.rank,n_spec,comm.size):
    fin=[]
    for j in range(n_avg):
        fin.append(fl[i*n_avg + j])
    print(fin)
    calc_pl(fin,out_prefix=out_prefix)
