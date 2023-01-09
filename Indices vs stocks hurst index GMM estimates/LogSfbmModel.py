#  Paper:   "From Rough to Multifractal volatility: the log S-fBM model"
#  Purpose: Data acquisition of the market data

# Author : Jean-Francois MUZY, Othmane ZARHALI

# Importations
import numpy as np
import pandas as pd


def make_basis_vector(size, index):
    arr = np.zeros(size)
    arr[index] = 1.0
    return arr

def ConstructBasis(size):
    return [make_basis_vector(size, i) for i in range(size) ]

def SumMultipleArrays(list_of_arrays):
   a = np.zeros(shape=list_of_arrays[0].shape) #initialize array of 0s
   for array in list_of_arrays:
      a += array
   return a


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
            u = np.random.normal(size=M)
            v = np.random.normal(size=M) * 1j

            thecorr = np.concatenate([covariance, np.flip(covariance)])
            if len(u) == len(thecorr)-2:
                thecorr = np.concatenate([covariance, np.flip(covariance[1:-1])])
            fftcorr = np.real(np.fft.fft(thecorr))

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
    def __init__(self,H=0.01,lambdasquare=0.02,T=0.61,sigma=1):  #  T=200,
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

    def GenerateSfbm(self,size=4096, subsample=8,t= None):
        if t==None:
            dt = 1 / subsample
        else:
            dt = t/(size * subsample)
        N = size - 1
        #N = size
        N = 2 ** np.ceil(np.log2(N))
        m, corr = self.SfbmCorrelation(size=N * subsample, dt=dt)
        log_mrm = GaussianProcess('LastWaveFFT',corr).Generate(size * subsample)
        om = log_mrm + m
        gg = np.random.normal(size=len(om))
        mrw = np.cumsum(np.exp(om) * gg * self.sigma * np.sqrt(dt))
        mrm = np.cumsum(self.sigma * self.sigma * np.exp(2 * om) * dt)
        mrw,mrm = mrw[::subsample],mrm[::subsample]
        #mrw = np.log(np.exp(mrw))
        return mrw,mrm


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
        zz2 = zz2 - np.mean(zz2)
        return zz1.values, zz2

    def Generate_sample(self,t,Delta,N_MC=100,size=4096,subsample=8, M=32,type_gen="ShiftedMRM"):
        sample = []
        if type_gen=="ShiftedMRM":
            for _ in range(N_MC):
                _, mrm = self.GenerateSfbm(size=size * M, subsample=subsample)
                mrm = mrm[t:int(t + Delta) + 1]
                #print("mean inside = ",np.mean(mrm),mrm)
                dmm = np.diff(mrm)
                dmm = np.append(dmm[0], dmm)
                sample.append(np.cumsum(dmm)[-1])
        if type_gen=="MRW sample":
            for _ in range(size):
                mrw, _ = self.GenerateSfbm(size=size * M, subsample=subsample,t=t)
                sample.append(mrw[-1])
        return sample


class MultidimensionalSfbm:
    def __init__(self,Sfbm_models,correlations={},dimension = 1,H_list=[],lambdasquare_list=[],T_list=[],sigma_list=[]):
        if type(Sfbm_models)!=list:
            TypeError("MultidimensionalSfbm error: Sfbm_models is not of expected type, list")
        # if any(type(Sfbm_model)!=Sfbm for Sfbm_model in Sfbm_models):
        #     TypeError("MultidimensionalSfbm error: Sfbm_models is not of expected type, Sfbm")
        All_lists = [H_list,lambdasquare_list,T_list,sigma_list]
        if (bool(correlations)==True) and any(len(item)!=dimension for item in All_lists):
            TypeError("MultidimensionalSfbm error: all list's size should be equal to dimension")
        else:
            self.Sfbm_models = Sfbm_models
            self.brownian_correlations = correlations
            if Sfbm_models == [] or  bool(correlations) == True :
                self.dimension = dimension
                self.correlation_flag = True
            else:
                self.correlation_flag = False
                self.dimension = len(Sfbm_models)
            #self.w_correlation_matrix = None   w will be iid
            self.H_list = H_list
            self.lambdasquare_list = lambdasquare_list
            self.T_list = T_list
            self.sigma_list = sigma_list


    def CorrelationMatrixBuilder_from_correlations(self, correlations):
        if type(correlations)!=dict:
            TypeError("MultidimensionalSfbm error: correlations should be a dictionary with keys of the form (i,j)")
        correl_matrix = [[0 for i in range(self.dimension)] for i in range(self.dimension)]
        for i in range(self.dimension):
            for j in range(i,self.dimension):
                if i==j:
                    correl_matrix[i][j] = 1
                else:
                    correl_matrix[i][j]=correlations[(i,j)]
        correl_matrix = np.array(correl_matrix)
        correl_matrix = correl_matrix + correl_matrix.T - np.diag(np.diag(correl_matrix))
        return correl_matrix

    def GenerateMultidimensionalSfbm(self,size=4096, subsample=8,brownian_correl_method = 'Brownian correlates - classical'):
        log_components = []
        if self.correlation_flag == False:
            if self.Sfbm_models == []:
                ValueError("MultidimensionalSfbm error: self.Sfbm_models should not be empty if correlation_flag =False")
            else:

                for Sfbm_model in self.Sfbm_models:
                    log_components.append(Sfbm_model.GenerateSfbm(size, subsample))
        else:
            log_mrm_matrix, dt = [], 1 / subsample
            # Independant w
            for dimension in range(self.dimension):
                Sfbm_model = Sfbm(self.H_list[dimension], self.lambdasquare_list[dimension], self.T_list[dimension],
                                  self.sigma_list[dimension])
                m, corr = Sfbm_model.SfbmCorrelation(size=(size - 1) * subsample, dt=dt)
                log_mrm = GaussianProcess('LastWaveFFT', corr).Generate(size * subsample)
                log_mrm = log_mrm + m
                log_mrm = log_mrm[::subsample]
                log_mrm_matrix.append(log_mrm + m)
            if brownian_correl_method == 'Brownian correlates - classical':
                brownian_correlation_matrix = self.CorrelationMatrixBuilder_from_correlations(self.brownian_correlations)
            if brownian_correl_method == 'Brownian correlates - random correl matrix':
                eigen_vectors =  ConstructBasis(self.dimension)   # canonical basis
                eigen_values = np.exp(log_mrm_matrix[0])  # w are independant
                brownian_correlation_matrix = SumMultipleArrays([eigen_values[i]*np.outer(eigen_vectors[i],eigen_vectors[i]) for i in range(self.dimension)])
            brownian_correlation_matrix_fact = np.linalg.cholesky(brownian_correlation_matrix)
            gg = np.random.normal(0, 1, (self.dimension, size))
            brownian_increments = np.array([brownian_correlation_matrix_fact @ gg[:, j] for j in range(size)]).T
            log_components = [(np.cumsum(np.exp(w) * brownian_increment * sigma * np.sqrt(dt)), np.array([])) for w, brownian_increment, sigma in zip(log_mrm_matrix, brownian_increments, self.sigma_list)]
        return log_components


    def Index_Builder(self,weights,trajectories, building_type = 'mrw'):
        import matplotlib.pyplot as plt
        # plt.plot(trajectories[0][0])
        # plt.title("Index_Builder - logprice")
        # plt.show()
        if len(weights)!= len(trajectories):
            ValueError("Sfbm error: weights and trajectories should be of the same length")
        if len(weights) != self.dimension:
            ValueError("Sfbm error: weights and dimension should be of the same length")
        else:
            index_mrw, index_mrm = 0,0
            if building_type == 'mrw':
                for i in range(self.dimension):
                    index_mrw += weights[i] * np.exp(trajectories[i][0])
                return np.log(index_mrw), np.array([])
            if building_type == 'mrm':
                for i in range(self.dimension):
                    index_mrm += weights[i] ** 2 * (trajectories[i][1])
                return np.array([]), index_mrm
            if building_type == 'mrm and mrw':
                for i in range(self.dimension):
                    index_mrm += weights[i] ** 2 * (trajectories[i][1])
                    index_mrw += weights[i] * np.exp(trajectories[i][0])
                return np.log(index_mrw), index_mrm


    def GeneratelogVolMultidimSfbm_Index(self,weights,method='quadratic variation estimate',size=4096, building_type = 'mrw',subsample=8,M=32,brownian_correl_method = 'Brownian correlates - classical'):  #,sigma=1, M=32
        # factor = 1
        # if self.H > 0:
        #     factor = M ** (-2 * self.H) / 4
        # self.lambdasquareintermittency *=  factor
        trajectories = self.GenerateMultidimensionalSfbm(size*M, subsample,brownian_correl_method)
        print("trajectories = ",trajectories)

        # if building_type == 'mrm':
        #     mrm_multidim_index = self.Index_Builder(weights, trajectories, building_type)
        # if building_type == 'mrw':
        #     mrw_multidim_index= self.Index_Builder(weights,trajectories,building_type)
        # else:
        #     mrw_multidim_index,mrm_multidim_index = self.Index_Builder(weights, trajectories, building_type)
        mrw_multidim_index, mrm_multidim_index = self.Index_Builder(weights, trajectories, building_type)

        # print("trajectories = ", trajectories)
        # print("mrw_multidim_index = ", mrw_multidim_index)

        # import matplotlib.pyplot as plt
        # plt.plot(mrw_multidim_index)
        # plt.title("GeneratelogVolMultidimSfbm_Index - mrw_multidim_index")
        # plt.show()

        # self.lambdasquareintermittency /= factor
        if self.correlation_flag == False:
            if method=='quadratic variation estimate':
                dvv = np.diff(np.array(mrw_multidim_index))
                dvv = np.append(dvv[0], dvv)
                quadratic_variation = np.cumsum(dvv * dvv)
                quadratic_variation = quadratic_variation[::M]
                quadratic_variation = quadratic_variation[1:] - quadratic_variation[:-1]
                logvol_index = np.log(quadratic_variation)
                logvol_index = pd.Series(logvol_index)
                logvol_index.replace([np.inf, -np.inf], np.nan, inplace=True)
                logvol_index = logvol_index.interpolate(limit_direction='both')
                logvol_index = logvol_index - logvol_index.mean()
                return logvol_index.values
            if method == 'direct':
                dmm = np.diff(mrm_multidim_index)
                dmm = np.append(dmm[0], dmm)
                mm = np.cumsum(dmm)
                mm = mm[::M]
                mm = mm[1:] - mm[:-1]
                logvol_index = np.log(mm)
                logvol_index = pd.Series(logvol_index)
                logvol_index.replace([np.inf, -np.inf], np.nan, inplace=True)
                logvol_index = logvol_index.interpolate(limit_direction='both')
                logvol_index = logvol_index - logvol_index.mean()
                return logvol_index.values
        if self.correlation_flag==True:
            if method == 'quadratic variation estimate':
                dvv = np.diff(np.array(mrw_multidim_index))
                dvv = np.append(dvv[0], dvv)
                quadratic_variation = np.cumsum(dvv * dvv)
                quadratic_variation = quadratic_variation[::M]
                quadratic_variation = quadratic_variation[1:] - quadratic_variation[:-1]
                logvol_index = np.log(quadratic_variation)
                logvol_index = pd.Series(logvol_index)
                logvol_index.replace([np.inf, -np.inf], np.nan, inplace=True)
                logvol_index = logvol_index.interpolate(limit_direction='both')
                logvol_index = logvol_index - logvol_index.mean()
                return logvol_index.values

class MultipleIndicesConstructor:
    def __init__(self,multiple_weights,multipleSfbm_models,multiple_correlations=[],multiple_Hs=[],multiple_lambdasquare_list=[],multiple_T_list=[],multiple_sigma_list=[]):
        if type(multiple_weights)!=list:
            TypeError("MultipleIndicesConstructor error: multiple_weights is not of expected type, list")
        if any(type(weights)!=list or type(weights)!=np.ndarray for weights in multiple_weights):
            TypeError("MultipleIndicesConstructor error: weights is not of expected type, list")
        if type(multipleSfbm_models) != list:
            TypeError("MultipleIndicesConstructor error: multipleSfbm_models is not of expected type, list")
        else:
            self.multiple_weights = multiple_weights
            self.multipleSfbm_models = multipleSfbm_models
            self.multiple_correlations = multiple_correlations
            self.multiple_Hs = multiple_Hs
            self.multiple_lambdasquare_list = multiple_lambdasquare_list
            self.multiple_T_list = multiple_T_list
            self.multiple_sigma_list = multiple_sigma_list
            self.correlation_flags = []

    def ConstructIndicestrajectories(self,size,subsample=4,building_type ='mrm and mrw',brownian_correl_method = 'Brownian correlates - classical'):
        Indicestrajectories = []
        # for weights, Sfbm_models,correlations,Hs,lambdasquare_list,T_list,sigma_list
        # in zip(*[self.multiple_weights,self.multipleSfbm_models, self.multiple_correlations,self.multiple_Hs,
        # self.multiple_lambdasquare_list,self.multiple_T_list,self.multiple_sigma_list]):
        list_index = 0
        for weights, Sfbm_models in zip(self.multiple_weights, self.multipleSfbm_models):
            # print("self.multiple_weights = ", self.multiple_weights)
            # print("weights = ", weights)
            # print("self.multiple_weights.index(weights) = ", self.multiple_weights.index(weights))
            #list_index = self.multiple_weights.index(weights)

            dimension = len(self.multiple_Hs[list_index])
            Index = MultidimensionalSfbm(Sfbm_models,self.multiple_correlations[list_index],dimension,self.multiple_Hs[list_index],self.multiple_lambdasquare_list[list_index],self.multiple_T_list[list_index],self.multiple_sigma_list[list_index])
            self.correlation_flags.append(Index.correlation_flag)
            Sfbms_generation_example = Index.GenerateMultidimensionalSfbm(size, subsample,brownian_correl_method)

            indices_trajectories = Index.Index_Builder(weights, Sfbms_generation_example,building_type)
            Indicestrajectories.append(indices_trajectories)

            list_index+=1
        return Indicestrajectories


    def ConstructLogVolIndicestrajectories(self,size,subsample=8,method='quadratic variation estimate',keys = [],M = 32,brownian_correl_method = 'Brownian correlates - classical',building_type = 'mrw'):

        log_vol_indices_trajectories,indices_trajectories = [],self.ConstructIndicestrajectories(size*M,subsample,building_type,brownian_correl_method)
        if keys != []:
            if len(indices_trajectories)!=len(keys):
                ValueError("MultipleIndicesConstructor error: keys and indices_trajectories should be of the same length")
        Multiple_indices_dic = dict()
        for i in range(len(indices_trajectories)):
            if self.correlation_flags[i] == False:
                if method == 'quadratic variation estimate':
                    dvv = np.diff(indices_trajectories[i][0])
                    dvv = np.append(dvv[0], dvv)
                    quadratic_variation = np.cumsum(dvv * dvv)
                    quadratic_variation = quadratic_variation[::M]
                    quadratic_variation = quadratic_variation[1:] - quadratic_variation[:-1]
                    logvol_index = np.log(quadratic_variation)

                    logvol_index = pd.Series(logvol_index)
                    logvol_index.replace([np.inf, -np.inf], np.nan, inplace=True)
                    logvol_index = logvol_index.interpolate(limit_direction='both')

                    logvol_index = logvol_index - logvol_index.mean()
                    log_vol_indices_trajectories.append(logvol_index)
                    #print("logvol_index = ",(logvol_index))
                    #print("indices_trajectories.index(index_trajectory) = ",indices_trajectories.index(index_trajectory))
                    Multiple_indices_dic[keys[i]+" "+str(i)] = logvol_index.values
                if method == 'direct':
                #for index_trajectory in indices_trajectories:
                    dmm = np.diff(indices_trajectories[i][1])
                    dmm = np.append(dmm[0], dmm)
                    mm = np.cumsum(dmm)
                    mm = mm[::M]
                    mm = mm[1:] - mm[:-1]
                    logvol_index = np.log(mm)
                    logvol_index = pd.Series(logvol_index)
                    logvol_index.replace([np.inf, -np.inf], np.nan, inplace=True)
                    logvol_index = logvol_index.interpolate(limit_direction='both')

                    logvol_index = logvol_index - logvol_index.mean()
                    log_vol_indices_trajectories.append(logvol_index)
                    Multiple_indices_dic[keys[i]+" "+str(i)] = logvol_index.values
            else:
                if method == 'quadratic variation estimate':
                    dvv = np.diff(indices_trajectories[i][0])
                    dvv = np.append(dvv[0], dvv)
                    quadratic_variation = np.cumsum(dvv * dvv)
                    quadratic_variation = quadratic_variation[::M]
                    quadratic_variation = quadratic_variation[1:] - quadratic_variation[:-1]
                    logvol_index = np.log(quadratic_variation)

                    logvol_index = pd.Series(logvol_index)
                    logvol_index.replace([np.inf, -np.inf], np.nan, inplace=True)
                    logvol_index = logvol_index.interpolate(limit_direction='both')

                    logvol_index = logvol_index - logvol_index.mean()
                    log_vol_indices_trajectories.append(logvol_index)
                    # print("logvol_index = ",(logvol_index))
                    # print("indices_trajectories.index(index_trajectory) = ",indices_trajectories.index(index_trajectory))
                    Multiple_indices_dic[keys[i] + " " + str(i)] = logvol_index.values
        if keys!=[]:
            return log_vol_indices_trajectories,Multiple_indices_dic
        else:
            return log_vol_indices_trajectories
