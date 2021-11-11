from numpy.lib.function_base import append
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy
from scipy import signal
import pywt as wt
from sklearn.decomposition import FastICA, PCA
from statsmodels.graphics.tsaplots import plot_acf, acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import json
import heartpy as hp
from scipy.signal import get_window
from parabolic import parabolic
from numpy import argmax, mean, diff, log, nonzero
from scipy.signal import blackmanharris, correlate, find_peaks, find_peaks_cwt
from scipy import signal as sg
from numpy import linalg as lg
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, minmax_scale
from sklearn.metrics import mean_squared_error, r2_score
from modwt import modwt, modwtmra
ID = []
HR = []
RR = []
T = []
HRref = []
RRref = []
Tref = []
HR_acf = []
RR_acf = []

HR_music = []
HR_root_music = []
RR_music = []
RR_root_music = []

HRref_music = []
RRref_music = []

HRref_peak = []
RRref_acf = []

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def compute_covariance(X):
    r"""This function estimate the covariance of a zero-mean numpy matrix. The covariance is estimated as :math:`\textbf{R}=\frac{1}{N}\textbf{X}\textbf{X}^{H}`
        
        
        :param X: M*N matrix
        :param type: string, optional
        :returns: covariance matrix of size M*M
        
        >>> import numpy as np
        >>> import spectral_analysis.spectral_analysis as sa
        >>> X = np.matrix('1 2; 3 4;5 6')
        >>> sa.compute_covariance(X)
        matrix([[  2.5,   5.5,   8.5],
        [  5.5,  12.5,  19.5],
        [  8.5,  19.5,  30.5]])
        
        """
        
    #Number of columns
    N=X.shape[1]
    R=(1./N)*X*X.H

    return R


def compute_autocovariance(x,M):
    
    r""" This function compute the auto-covariance matrix of a numpy signal. The auto-covariance is computed as follows
        
        .. math:: \textbf{R}=\frac{1}{N}\sum_{M-1}^{N-1}\textbf{x}_{m}\textbf{x}_{m}^{H}
        
        where :math:`\textbf{x}_{m}^{T}=[x[m],x[m-1],x[m-M+1]]`.
        
        :param x: ndarray of size N
        :param M:  int, optional. Size of signal block.
        :returns: ndarray
        
        """
    
    # Create covariance matrix for psd estimation
    # length of the vector x
    N=x.shape[0]
    
    #Create column vector from row array
    x_vect=np.transpose(np.matrix(x))
    
    # init covariance matrix
    yn=x_vect[M-1::-1]
    R=yn*yn.H
    for indice in range(1,N-M):
        #extract the column vector
        yn=x_vect[M-1+indice:indice-1:-1]
        R=R+yn*yn.H
    
    R=R/N
    return R


def pseudospectrum_MUSIC(x,L,M=None,Fe=1,f=None):
    r""" This function compute the MUSIC pseudospectrum. The pseudo spectrum is defined as
        
        .. math:: S(f)=\frac{1}{\|\textbf{G}^{H}\textbf{a}(f) \|}
        
        where :math:`\textbf{G}` corresponds to the noise subspace and :math:`\textbf{a}(f)` is the steering vector. The peek locations give the frequencies of the signal.
        
        :param x: ndarray of size N
        :param L: int. Number of components to be extracted.
        :param M:  int, optional. Size of signal block.
        :param Fe: float. Sampling Frequency.
        :param f: nd array. Frequency locations f where the pseudo spectrum is evaluated.
        :returns: ndarray
        
        >>> from pylab import *
        >>> import numpy as np
        >>> import spectral_analysis.spectral_analysis as sa
        >>> Fe=500
        >>> t=1.*np.arange(100)/Fe
        >>> x=np.exp(2j*np.pi*55.2*t)
        >>> f,P=sa.pseudospectrum_MUSIC(x,1,100,Fe,None)
        >>> plot(f,P)
        >>> show()
        
        """
    
    # length of the vector x
    N=x.shape[0]
    
    if np.any(f)==None:
        f=np.linspace(0.,Fe//2,1024)

    if M==None:
        M=N//2

    #extract noise subspace
    R=compute_autocovariance(x,M)
    U,S,V=lg.svd(R)
    G=U[:,L:]

    #compute MUSIC pseudo spectrum
    N_f=f.shape
    cost=np.zeros(N_f)
    
    for indice,f_temp in enumerate(f):
        # construct a (note that there a minus sign since Yn are defined as [y(n), y(n-1),y(n-2),..].T)
        vect_exp=-2j*np.pi*f_temp*np.arange(0,M)/Fe
        a=np.exp(vect_exp)
        a=np.transpose(np.matrix(a))
        #Cost function
        cost[indice]=1./lg.norm((G.H)*a)

    return f,cost
def root_MUSIC(x,L,M,Fe=1):
    
    r""" This function estimate the frequency components based on the roots MUSIC algorithm [BAR83]_ . The roots Music algorithm find the roots of the following polynomial
        
        .. math:: P(z)=\textbf{a}^{H}(z)\textbf{G}\textbf{G}^{H}\textbf{a}(z)
        
        The frequencies are related to the roots as 
        
        .. math:: z=e^{-2j\pi f/Fe}
        
        :param x: ndarray of size N
        :param L: int. Number of components to be extracted.
        :param M:  int, optional. Size of signal block.
        :param Fe: float. Sampling Frequency.
        :returns: ndarray containing the L frequencies
        
        >>> import numpy as np
        >>> import spectral_analysis.spectral_analysis as sa
        >>> Fe=500
        >>> t=1.*np.arange(100)/Fe
        >>> x=np.exp(2j*np.pi*55.2*t)
        >>> f=sa.root_MUSIC(x,1,None,Fe)
        >>> print(f)
        """

    # length of the vector x
    N=x.shape[0]
    
    if M==None:
        M=N//2
    
    #extract noise subspace
    R=compute_autocovariance(x,M)
    U,S,V=lg.svd(R)
    G=U[:,L:]

    #construct matrix P
    P=G*G.H

    #construct polynomial Q
    Q=0j*np.zeros(2*M-1)
    #Extract the sum in each diagonal
    for (idx,val) in enumerate(range(M-1,-M,-1)):
        diag=np.diag(P,val)
        Q[idx]=np.sum(diag)

    #Compute the roots
    roots=np.roots(Q)

    #Keep the roots with radii <1 and with non zero imaginary part
    roots=np.extract(np.abs(roots)<1,roots)
    roots=np.extract(np.imag(roots) != 0,roots)

    #Find the L roots closest to the unit circle
    distance_from_circle=np.abs(np.abs(roots)-1)
    index_sort=np.argsort(distance_from_circle)
    component_roots=roots[index_sort[:L]]

    #extract frequencies ((note that there a minus sign since Yn are defined as [y(n), y(n-1),y(n-2),..].T))
    angle=-np.angle(component_roots)

    #frequency normalisation
    f=Fe*angle/(2.*np.pi)

    return f
for num in range(0, 900):
    try:
        Heart = pd.read_csv(str(num) + "--Heartrate.csv", delimiter=",")
        '''
        #pulse = Heart['Wavelet-3']
        raw_pulse = minmax_scale(Heart['Raw'])
        raw_pulse = raw_pulse[30:]
        raw_pulse = pd.Series(raw_pulse)
        Xp = raw_pulse.index
        Xp = np.reshape(Xp, (len(Xp), 1))
        pfp = PolynomialFeatures(degree=5)
        Xpp = pfp.fit_transform(Xp)
        md2p = LinearRegression()
        md2p.fit(Xpp, raw_pulse)
        trendpp = md2p.predict(Xpp) 

        detrpolyp = [raw_pulse[i] - trendpp[i] for i in range(0, len(raw_pulse))] 
        wt = modwt(raw_pulse, 'sym2', 4)
        wtmra = modwtmra(wt, 'sym2')

        #pulse_w3 = minmax_scale(wtmra[3])
        #pulse_w3 = wtmra[3]
        '''
        pulse_w3 = Heart['Wavelet-3']
        
        #-------Autocorrelation---------------------------------
        corr = correlate(pulse_w3, pulse_w3, mode='full')
        corr = corr[len(corr)//2:]
        d = diff(corr)
        #d = d[30:270]
        freq_acf = root_MUSIC(d, 1, M=None, Fe=30)
        freq_acf = round(abs(freq_acf[0]*60),2)
        HR_acf.append(freq_acf)
        '''
        #------MUSIC--------------------------
        f,P= pseudospectrum_MUSIC(pulse_w3, 1, M=None, Fe=30)
        HR_music.append((f[np.argmax(P)]*60))
        estimated_hr = round((f[np.argmax(P)]*60))
        '''
        #---root MUSIC ----------------------------------------
        freq = root_MUSIC(pulse_w3, 1, M=None, Fe=30)
        freq = round(abs(freq[0]*60),2)
        HR_root_music.append(freq)


        ID.append(num)
        Signals,axs = plt.subplots(6, 1,figsize=(10,20))
        axs[0].plot(Heart['Raw'], 'g')
        #axs[0].plot(pulse_trend_x)
        #axs[0].plot(pulse_trend_y)
        #axs[0].plot(raw_pulse_detrended,'b')
        #axs[0].plot(trendpp) 
        #axs[0].plot(detrpolyp) 

        d = minmax_scale(d)
        pulse_w3 = minmax_scale(pulse_w3)
        axs[1].plot(d,'r')
        axs[1].plot(pulse_w3,'g')
        #axs[1].plot(pulse_w3_full,'g')
        #axs[1].set_xlim(0, 300)
        axs[1].text(0, 0.2, 'HR = ' + str(freq), horizontalalignment='left', verticalalignment='top', fontsize=28, color='C1', transform = axs[1].transAxes)

        axs[1].text(0, 0.5, 'HR = ' + str(freq_acf), horizontalalignment='left', verticalalignment='top', fontsize=28, color='C2', transform = axs[1].transAxes)

        try:
            Resp = pd.read_csv(str(num) + "--Respiratory.csv", delimiter=",")
            vol = Resp['Raw']
            min_dist = Resp['Dist']
            min = vol.min()
        
            vol = np.where(vol > (min + 2000), (min - 100),vol)
            vol = pd.Series(vol)
            med = vol.median()
            vol = np.where(vol < min, med,vol)
            #axs[3].plot(vol)
            #axs[3].set_xlim(0, 300)
            vol = hp.filter_signal(vol, [0.1, 0.6], sample_rate=30, order=3, filtertype='bandpass')
            #corr = correlate(vol, vol, mode='full')
            #corr = corr[len(corr)//2:]
            #vol = moving_average(vol,30)
            #vol = diff(vol)
            vol = pd.Series(vol)
            X = vol.index
            X = np.reshape(X, (len(X), 1))
            min_val = abs(vol.min())
            Max_val = abs(vol.max())
            if min_val < Max_val:
                mM_Diff = Max_val/min_val
            else:
                mM_Diff = min_val/Max_val
            if mM_Diff <= 1.5:
                deg = 1
            else:
                deg = 2
            pf = PolynomialFeatures(degree = 2)#deg)
            Xp = pf.fit_transform(X)
            md2 = LinearRegression()
            md2.fit(Xp, vol)
            trendp = md2.predict(Xp) 

            axs[3].plot(trendp) 
            axs[3].plot(vol)
            axs[3].set_xlim(0, 300)
            detrpoly = [vol[i] - trendp[i] for i in range(0, len(vol))]  
            #corr = correlate(detrpoly, detrpoly, mode='full')
            #detrpoly = corr[len(corr)//2:]
            #vol = pd.Series(vol)
            detrpoly = pd.Series(detrpoly)
            #corr = correlate(detrpoly, detrpoly, mode='full')
            #corr = corr[len(corr)//2:]
            
            #detrpoly = diff(detrpoly)
            #vol = hp.filter_signal(vol, [0.1, 1], sample_rate=30, order=1, filtertype='bandpass')

            #detrpoly = detrpoly.diff(periods = 30)
            #detrpoly = detrpoly[30:] 
            #detrpoly = hp.filter_signal(detrpoly, [0.1, 1], sample_rate=30, order=1, filtertype='bandpass')
            
            
            #corr_crop = diff(corr)#[30:]
            '''
            #-------MUSIC----------------------------------------------------
            fR,PR= pseudospectrum_MUSIC(detrpoly, 1, M=None, Fe=30)
            estimated_rr = round(fR[np.argmax(PR)]*60)
            RR_music.append(fR[np.argmax(PR)]*60)
            estimated_rr = round(fR[np.argmax(PR)]*60)
            '''
             #-------Autocorrelation---------------------------------
            corr_RR = correlate(detrpoly, detrpoly, mode='full')
            corr_RR = corr_RR[len(corr_RR)//2:]
            d_RR = diff(corr_RR)
            #d = d[30:270]
            freq_acf_RR = root_MUSIC(d_RR, 1, M=None, Fe=30)
            freq_acf_RR = round(abs(freq_acf_RR[0]*60),2)
            RR_acf.append(freq_acf_RR)

            #---root MUSIC ----------------------------------------
            freq_rr = root_MUSIC(detrpoly, 1, M=None, Fe=30)
            freq_rr = round(abs(freq_rr[0]*60),2)
            RR_root_music.append(freq_rr)

            axs[4].text(0, 0.2, 'RR = ' + str(freq_rr), horizontalalignment='left', verticalalignment='top', fontsize=28, color='C1', transform = axs[4].transAxes)
            
            axs[4].text(0, 0.5, 'RR = ' + str(freq_acf_RR), horizontalalignment='left', verticalalignment='top', fontsize=28, color='C2', transform = axs[4].transAxes)
           

            #plot_pacf(detrpoly)
            #plt.show()
            detrpoly = minmax_scale(detrpoly)
            d_RR = minmax_scale(d_RR)
            axs[4].plot(detrpoly,'g')
            axs[4].plot(d_RR,'r')
            #axs[4].plot(peaks,corr[peaks],'x')
            axs[4].set_xlim(0, 300)
        except Exception as ex:
            print('index not found RR -' + str(num))
            RR.append(0)
            
        try:
            Temp = pd.read_csv(str(num) + "--Temperature.csv", delimiter=",")
            T.append(round(Temp['Temp'].mean(),2))
        except Exception as ex:
            print('index not found T -' + str(num))
            T.append(0)
        try:
        #----- Reference Heartrate--------------------------------------------------------------------
            Ref = pd.read_csv(str(num) + "--Reference.csv", delimiter=",")
            pulse1 = hp.filter_signal(Ref['H'], [0.6, 2.5], sample_rate=30, order=3, filtertype='bandpass')

            #fr,Pr= pseudospectrum_MUSIC(pulse1, 1, M=None, Fe=30)
            freq_rhr = root_MUSIC(pulse1, 1, M=None, Fe=30)
            if num == 10 or num == 13 or num == 14 or num == 82 or num == 83 or num == 84:
                HRref_music.append(0)
                ref_hr = 0

            else:
                #HRref_music.append(fr[np.argmax(Pr)]*60)
                ref_hr = round(abs(freq_rhr[0]*60))
                HRref_music.append(abs(freq_rhr[0]*60))
            axs[2].plot(pulse1)
            axs[2].text(0, 0.9, 'HR = ' + str(ref_hr), horizontalalignment='left', verticalalignment='top', fontsize=28, color='C1', transform = axs[2].transAxes)
            

        #----- Reference Respiratoty--------------------------------------------------------------------
            vol1 = hp.filter_signal(Ref['R'], [0.1, 1], sample_rate=30, order=1, filtertype='bandpass')
           
            #frr,Prr= pseudospectrum_MUSIC(vol1, 1, M=None, Fe=30)
            freq_rrr = root_MUSIC(vol1, 1, M=None, Fe=30)
            RRref_music.append(abs(freq_rrr[0]*60))
            ref_rr = round(abs(freq_rrr[0]*60))

            axs[5].plot(vol1)
            axs[5].set_xlim(0, 300)
            axs[5].text(0, 0.9, 'RR = ' + str(ref_rr), horizontalalignment='left', verticalalignment='top', fontsize=28, color='C1', transform = axs[5].transAxes)
        except Exception as ex:
            print('index not found REF -' + str(num))
            HRref.append(0)
            HRref_music.append(0)
            RRref.append(0)
            RRref_music.append(0)
            HRref_peak.append(0)
            RRref_acf.append(0)
        plt.savefig(str(num)+'.png')
        plt.close()
    except Exception as ex:
        print('index not found -' + str(num))
    
sI = pd.Series(ID, name='idx')
#sH = pd.Series(HR, name='hr')
#sR = pd.Series(RR, name='rr')
sT = pd.Series(T, name='t')
#srH = pd.Series(HRref, name='hrRef')
##srR = pd.Series(RRref, name='rrRef')
srT = pd.Series(Tref, name='tRef')

sHacf = pd.Series(HR_acf, name='hr_acf')
sRacf = pd.Series(RR_acf, name='rr_acf')

sHmusic = pd.Series(HR_music, name='hr_music')
sHrootmusic = pd.Series(HR_root_music, name='hr_root_music')
sRmusic = pd.Series(RR_music, name='rr_music')
sRrootmusic = pd.Series(RR_root_music, name='rr_root_music')


srHmusic = pd.Series(HRref_music, name='hrRef_music')
srRmusic = pd.Series(RRref_music, name='rrRef_music')

#srHpeak = pd.Series(HRref_peak, name='hrRef_peak')

#df = pd.concat([sI, sH, sHacf, sHmusic, srH, srHmusic, srHpeak, sR, sRacf, sRmusic, srR, srRmusic, sT, srT], axis=1)
df = pd.concat([sI, sHacf, sHrootmusic, srHmusic, sRacf, sRrootmusic, srRmusic, sT, srT], axis=1)

df.to_csv('DataCov.csv')

'''
num = 6
Heart = pd.read_csv(str(num) + "--Heartrate.csv", delimiter=",")
Ref = pd.read_csv(str(num) + "--Reference.csv", delimiter=",")
Resp = pd.read_csv(str(num) + "--Respiratory.csv", delimiter=",")
Temp = pd.read_csv(str(num) + "--Temperature.csv", delimiter=",")
Signals,axs = plt.subplots(4, 1,figsize=(6,3))
Signals.subplots_adjust(top=0.92, bottom=0.13, left=0.10, right=0.95, hspace=0.6,wspace=0.35)

#pulse = hp.filter_signal(Ref['H'], [0.6, 3], sample_rate=30, order=1, filtertype='bandpass')
pulse = Heart['Wavelet-3']
axs[0].plot(Ref['H'])
axs[1].plot(Heart['Wavelet-3'])
axs[2].plot(Ref['R'])
axs[3].plot(Resp['Bandpass'])

print('Temperature = ' + str(round(Temp['Temp'].mean(),1)) + ' C')


w = get_window('blackmanharris', len(pulse))
#pulse = signal.resample(Ref['H'], 300)
yf=np.fft.fft((pulse/len(pulse)))
yf=yf[range(int(len(pulse)/2.0))]
tc=len(pulse)
val=np.arange(int(tc/2.0))
tp=tc/30.0
fr=val/tp
print('Heartrate = ' + str(round(60.0*fr[np.argmax(abs(yf))],1)) + ' beat per min')
plt.figure(2)
plt.plot (val, abs(yf), "r")


w = np.fft.rfft(pulse * w, n=len(pulse))
freqs = np.fft.rfftfreq(len(pulse), d=0.033)
plt.figure(3)
plt.plot(freqs, 20*np.log10(np.abs(w)))
plt.ylim(-60, 60)
plt.xlim(0, 3)


#axs[4].set_xlim(0,400)
f = signal.resample(Resp['Bandpass'], 100)
w1 = get_window('blackmanharris', len(Resp['Bandpass']))
yf1=np.fft.fft((Resp['Bandpass']/len(Resp['Bandpass'])))
print(yf1.shape)
yf1=yf1[range(int(len(Resp['Bandpass'])/2.0))]

tc1=len(Resp['Bandpass'])
val1=np.arange(int(tc1/2.0))
tp1=tc1/30.0
fr1=val1/tp1
#f = signal.resample(Resp['Bandpass'], 100)
print('Respiratory = ' + str(round(60.0
*fr1[np.argmax(abs(yf1))],1)) + ' breath per min')

w2 = np.fft.rfft(Resp['Bandpass'] , n=len(Resp['Bandpass']))
freqs = np.fft.rfftfreq(len(Resp['Bandpass']), d=0.033)

print('Respiratory = ' + str(round(60.0
*freqs[np.argmax(abs(w2))],1)) + ' breath per min')
plt.figure(4)
plt.plot(freqs, 20*np.log10(np.abs(w2)))
plt.ylim(-60, 60)
plt.xlim(0, 3)

plt.show()
'''