#  Paper:   "From Rough to Multifractal volatility: the log S-fBM model"
#  Purpose: Data acquisition of the market data

# Author : Jean-Francois MUZY, Othmane ZARHALI

# Importations
import numpy as np
import pandas as pd



class GaussianProcess:
    def __init__(self,generation_type,covariance_matrix):
        if type(generation_type)!= str:
            TypeError("GaussianProcess error: generation_type is not of expected type, str")
        if type(covariance_matrix)!= np.ndarray:
            TypeError("GaussianProcess error: covariance is not of expected type, np.ndarray")
        else:
            self.generation_type = generation_type
            self.covariance_matrix = covariance_matrix

    def Generate(self,size):
        if self.generation_type == 'LastWaveFFT':
            m = int(2 ** np.floor(np.log2(size)))
            M = 2 * m
            M=2*len(self.covariance_matrix)
            covariance = self.covariance_matrix
            # zero padding
            if m + 1 - len(self.covariance_matrix)>0:
                covariance = np.concatenate([self.covariance_matrix, np.zeros(m + 1 - len(self.covariance_matrix))])
            #thecorr = np.concatenate([covariance, np.flip(covariance[1:-1])])
            thecorr = np.concatenate([covariance, np.flip(covariance)])
            fftcorr = np.real(np.fft.fft(thecorr))
            u = np.random.normal(size=M)
            v = np.random.normal(size=M) * 1j
            fftcorr = np.sqrt(fftcorr + 0j) * (u + v) / np.sqrt(2)
            corr = np.real(np.fft.ifft(fftcorr))
            return corr[:size] * np.sqrt(2 * M)
        # if self.generation_type == 'LastWaveFFT':
        #     m = int(2 ** np.ceil(np.log2(size - 1)))
        #     M = 2 * m
        #     if (not flagZero) and (len(self.covariance_matrix) < m):
        #         print("Sorry the size {} of the autocorrelation signal must be strictly greater than {}".format(
        #             len(self.covariance_matrix), m))
        #         # return -1
        #     if (flagZero & (len(self.covariance_matrix) <= m)):
        #         covariance = np.concatenate([[self.covariance_matrix, np.zeros(m + 1 - len(self.covariance_matrix))]])
        #     else:
        #         covariance = self.covariance_matrix
        #     thecorr = np.concatenate([[covariance[0:m + 1], np.flip(covariance[1:m])]])
        #     fftcorr = np.real(np.fft.fft(thecorr))
        #     u = np.random.normal(size=M)
        #     v = np.random.normal(size=M) * 1j
        #     fftcorr = np.sqrt(fftcorr + 0j) * (u + v) / np.sqrt(2)
        #     corr = np.real(np.fft.ifft(fftcorr))
        #     return corr[:size] * np.sqrt(2 * M)


class Sfbm:
    def __init__(self,H=0,lambdasquare=0.02):  #  T=200,
        self.H=H
        self.lambdasquareintermittency = lambdasquare

    def SfbmCorrelation(self,size,T, dt=.1):
        tau = dt * np.arange(1, size)
        if self.H == 0:
            correlation_list = self.lambdasquareintermittency * np.log(T / tau) * (tau < T)
            correlation_list = np.append(self.lambdasquareintermittency * (1 + np.log(T / dt)), correlation_list)
        else:
            xx = np.append(0, tau)
            K = self.lambdasquareintermittency / (2 * self.H * (1 - 2 * self.H))
            correlation_list = K * ((T * dt) ** (2 * self.H) - xx ** (2 * self.H)) * (xx < T * dt)
        m = -correlation_list[0]
        return m, correlation_list

    def GenerateSfbm(self,size=4096,T=200, subsample=4, sigma=1):
        dt = 1 / subsample
        N = size - 1
        N = 2 ** np.ceil(np.log2(N))
        m, corr = self.SfbmCorrelation(size=N * subsample,T=T, dt=dt)
        log_mrm = GaussianProcess('LastWaveFFT',corr).Generate(size * subsample)
        # print("m = ",m)  #,len(m)
        # print("log_mrm = ", log_mrm, len(log_mrm))
        om = log_mrm + m
        gg = np.random.normal(size=len(om))
        mrw = np.cumsum(np.exp(om) * gg * sigma * np.sqrt(dt))
        mrm = np.cumsum(sigma * sigma * np.exp(2 * om) * dt)
        return mrw[::subsample], mrm[::subsample]

    def GeneratelogVol(self,T,size=4096, subsample=8,sigma=1, M=32):
        factor = 1
        if self.H > 0:
            factor = M ** (-2 * self.H) / 4
        self.lambdasquareintermittency *=  factor
        mrw, mrm = self.GenerateSfbm(size=size * M, T=T,subsample=subsample,sigma=sigma)
        self.lambdasquareintermittency /= factor
        dvv = np.diff(np.array(mrw))
        dvv = np.append(dvv[0], dvv)
        dmm = np.diff(mrm)
        dmm = np.append(dmm[0], dmm)

        qv = np.cumsum(dvv * dvv)
        mm = np.cumsum(dmm)
        qv = qv[::M]
        mm = mm[::M]
        qv = qv[1:] - qv[:-1]

        mm = mm[1:] - mm[:-1]
        zz1 = np.log(qv)
        zz1 = pd.Series(zz1)
        zz1.replace([np.inf, -np.inf], np.nan, inplace=True)
        zz1 = zz1.interpolate(limit_direction='both')
        zz1 = zz1 - zz1.mean()

        zz2 = np.log(mm) - np.mean(np.log(mm))
        return zz1.values, zz2

