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
            # M = 2 * m
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
    def __init__(self,H=0.01,lambdasquare=0.02,T=200,sigma=1):  #  T=200,
        self.H=H
        self.lambdasquareintermittency = lambdasquare
        self.T = T
        self.sigma = sigma

    def SfbmCorrelation(self,size, dt=.1):
        tau = dt * np.arange(1, size)
        if self.H == 0:
            correlation_list = self.lambdasquareintermittency * np.log(self.T / tau) * (tau < self.T)
            correlation_list = np.append(self.lambdasquareintermittency * (1 + np.log(self.T / dt)), correlation_list)
        else:
            xx = np.append(0, tau)
            K = self.lambdasquareintermittency / (2 * self.H * (1 - 2 * self.H))
            correlation_list = K * ((self.T * dt) ** (2 * self.H) - xx ** (2 * self.H)) * (xx < self.T * dt)
        m = -correlation_list[0]
        return m, correlation_list

    def GenerateSfbm(self,size=4096, subsample=4):
        dt = 1 / subsample
        N = size - 1
        #N = size
        N = 2 ** np.ceil(np.log2(N))
        m, corr = self.SfbmCorrelation(size=N * subsample, dt=dt)
        log_mrm = GaussianProcess('LastWaveFFT',corr).Generate(size * subsample)
        #print("log_mrm log_vol_index= ",log_mrm)
        om = log_mrm + m
        gg = np.random.normal(size=len(om))
        mrw = np.cumsum(np.exp(om) * gg * self.sigma * np.sqrt(dt))
        mrm = np.cumsum(self.sigma * self.sigma * np.exp(2 * om) * dt)
        return mrw[::subsample], mrm[::subsample]


    def GeneratelogVol(self,size=4096, subsample=8, M=32):
        factor = 1
        if self.H > 0:
            factor = M ** (-2 * self.H) / 4
        self.lambdasquareintermittency *=  factor
        mrw, mrm = self.GenerateSfbm(size=size * M,subsample=subsample)
        self.lambdasquareintermittency /= factor
        dvv = np.diff(np.array(mrw))
        dvv = np.append(dvv[0], dvv)
        dmm = np.diff(mrm)
        dmm = np.append(dmm[0], dmm)

        quadratic_variation = np.cumsum(dvv * dvv)
        mm = np.cumsum(dmm)
        quadratic_variation = quadratic_variation[::M]
        mm = mm[::M]
        quadratic_variation = quadratic_variation[1:] - quadratic_variation[:-1]

        mm = mm[1:] - mm[:-1]
        zz1 = np.log(quadratic_variation)
        zz1 = pd.Series(zz1)
        zz1.replace([np.inf, -np.inf], np.nan, inplace=True)
        zz1 = zz1.interpolate(limit_direction='both')
        zz1 = zz1 - zz1.mean()
        zz2 = np.log(mm)
        zz2 = zz2 - np.mean(np.log(mm))

        return zz1.values, zz2


class MultidimensionalSfbm:
    def __init__(self,Sfbm_models,correlation=0):
        if type(Sfbm_models)!=list:
            TypeError("MultidimensionalSfbm error: Sfbm_models is not of expected type, list")
        if any(type(Sfbm_model)!=Sfbm for Sfbm_model in Sfbm_models):
            TypeError("MultidimensionalSfbm error: Sfbm_models is not of expected type, Sfbm")
        else:
            self.Sfbm_models = Sfbm_models
            self.correlation = correlation

    def GenerateMultidimensionalSfbm(self,size=4096, subsample=4):
        Generation_list = []
        if self.correlation==0:
            for Sfbm_model in self.Sfbm_models:
                Generation_list.append(Sfbm_model.GenerateSfbm(size, subsample))
            return Generation_list

    def Index_Builder(self,weights,trajectories, building_type = 'mrw'):
        if len(weights)!= len(trajectories):
            ValueError("Sfbm error: weights and trajectories should be of the same length")
        else:
            dimension = len(weights)
            index_mrw,index_mrm = 0,0

            if building_type=='mrw':
                print('888888888888888888888888888')
                for i in range(dimension):
                    index_mrw+=weights[i] * trajectories[i][0]
                return index_mrw
            if building_type=='mrm':
                print('888888888888888888888888888')
                for i in range(dimension):
                    index_mrm += weights[i]**2*(trajectories[i][1])
                return index_mrm
            if building_type=='mrm and mrw':
                print('888888888888888888888888888')
                for i in range(dimension):
                    index_mrm += weights[i]**2*(trajectories[i][1])
                    index_mrw += weights[i] * trajectories[i][0]
                return index_mrw,index_mrm

    def GeneratelogVolMultidimSfbm_Index(self,weights,method='quadratic variation estimate',size=4096, subsample=8):  #,sigma=1, M=32
        # factor = 1
        # if self.H > 0:
        #     factor = M ** (-2 * self.H) / 4
        # self.lambdasquareintermittency *=  factor
        trajectories = self.GenerateMultidimensionalSfbm(size, subsample)
        mrw_multidim_index,mrm_multidim_index= self.Index_Builder(weights,trajectories,'mrm and mrw')
        # self.lambdasquareintermittency /= factor
        if method=='quadratic variation estimate':
            dvv = np.diff(np.array(mrw_multidim_index))
            dvv = np.append(dvv[0], dvv)

            quadratic_variation = np.cumsum(dvv * dvv)
            logvol_index = np.log(quadratic_variation)
            logvol_index = pd.Series(logvol_index)
            logvol_index.replace([np.inf, -np.inf], np.nan, inplace=True)
            logvol_index = logvol_index.interpolate(limit_direction='both')
            logvol_index = logvol_index - logvol_index.mean()
            return logvol_index.values
        if method=='direct':
            logvol_index = pd.Series(np.log(mrm_multidim_index))
            logvol_index.replace([np.inf, -np.inf], np.nan, inplace=True)
            logvol_index = logvol_index.interpolate(limit_direction='both')
            logvol_index = logvol_index - logvol_index.mean()
            return logvol_index.values

class MultipleIndicesConstructor:
    def __init__(self,multiple_weights,multipleSfbm_models):
        if type(multiple_weights)!=list:
            TypeError("MultipleIndicesConstructor error: multiple_weights is not of expected type, list")
        if any(type(weights)!=list for weights in multiple_weights):
            TypeError("MultipleIndicesConstructor error: weights is not of expected type, list")
        if type(multipleSfbm_models) != list:
            TypeError("MultipleIndicesConstructor error: multipleSfbm_models is not of expected type, list")
        # if any(type(Sfbm_models) != Sfbm for Sfbm_models in multipleSfbm_models):
        #     TypeError("MultipleIndicesConstructor error: Sfbm_models is not of expected type, Sfbm")
        # if any(len(weights)!=len(Sfbm_models) for weights,Sfbm_models in zip(self.multiple_weights,self.multipleSfbm_models)):
        #     ValueError("MultipleIndicesConstructor error: Sfbm_models and weights should be of the same length")
        else:
            self.multiple_weights = multiple_weights
            self.multipleSfbm_models = multipleSfbm_models

    def ConstructIndicestrajectories(self,size,subsample=4):
        indices_trajectories = []
        for weights,Sfbm_models in zip(self.multiple_weights,self.multipleSfbm_models):
            index = MultidimensionalSfbm(Sfbm_models)
            multidimensional_trajectories = index.GenerateMultidimensionalSfbm(size,subsample)
            index_mrw, index_mrm = index.Index_Builder(weights,multidimensional_trajectories, 'mrw and mrm')
            print("index_mrw, index_mrm = ",index_mrw, index_mrm)
            indices_trajectories.append(index_mrw, index_mrm)
        return indices_trajectories

    def ConstructLogVolIndicestrajectories(self,size,subsample=4,method='quadratic variation estimate'):
        log_vol_indices_trajectories,indices_trajectories = [],self.ConstructIndicestrajectories(size,subsample)
        print("indices_trajectories = ",indices_trajectories)
        if method=='quadratic variation estimate':
            for index_trajectory in indices_trajectories:
                dvv = np.diff(index_trajectory[0])
                dvv = np.append(dvv[0], dvv)
                quadratic_variation = np.cumsum(dvv * dvv)
                logvol_index = np.log(quadratic_variation)
                logvol_index = pd.Series(logvol_index)
                logvol_index.replace([np.inf, -np.inf], np.nan, inplace=True)
                logvol_index = logvol_index.interpolate(limit_direction='both')
                logvol_index = logvol_index - logvol_index.mean()
                log_vol_indices_trajectories.append(logvol_index)
        if method == 'direct':
            for index_trajectory in indices_trajectories:
                logvol_index = pd.Series(np.log(index_trajectory[1]))
                logvol_index.replace([np.inf, -np.inf], np.nan, inplace=True)
                logvol_index = logvol_index.interpolate(limit_direction='both')
                logvol_index = logvol_index - logvol_index.mean()
                log_vol_indices_trajectories.append(logvol_index)
        return log_vol_indices_trajectories
