#  Paper:   "From Rough to Multifractal volatility: the log S-fBM model"
#  Purpose: Data acquisition of the market data

# Author : Jean-Francois MUZY, Othmane ZARHALI

# Importations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from numba import njit
import seaborn as sns

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
            # print("len(fftcorr),len(u),len(v) = ",len(fftcorr),len(fftcorr[0]),len(u),len(v))
            # print("np.sqrt(fftcorr + 0j) * (u + v) = ",np.sqrt(fftcorr + 0j) * np.transpose((u + v)))


            if len([fftcorr[0]])!=len(u):
                fftcorr = np.transpose(np.transpose(np.sqrt(fftcorr + 0j)) * (u + v) / np.sqrt(2))
            else:
                fftcorr = np.sqrt(fftcorr + 0j) * (u + v) / np.sqrt(2)

            # fftcorr = np.sqrt(fftcorr + 0j) * (u + v) / np.sqrt(2)
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
        else:
            return None


class Sfbm:
    def __init__(self,H=0.01,lambdasquare=0.02,T=0.61,sigma=1):  #  T=200,
        self.H=H
        self.lambdasquare = lambdasquare
        self.T = T
        self.sigma = sigma

    def SfbmCorrelation(self,size, dt=.1):
        timesteps = dt * np.arange(1, size)
        if self.H == 0:
            correlation_list = self.lambdasquare * np.log(self.T / timesteps) * (timesteps < self.T)
            correlation_list = np.append(self.lambdasquare * (1 + np.log(self.T / dt)), correlation_list)
        else:
            xx = np.append(0, timesteps)
            correlation_list = self.lambdasquare / (2 * self.H * (1 - 2 * self.H)) * ((self.T * dt) ** (2 * self.H) - xx ** (2 * self.H)) * (xx < self.T * dt)
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


    def GeneratelogVol(self,size=4096, subsample=8, M=32,t=None):
        factor = 1
        if self.H > 0:
            factor = M ** (-2 * self.H) / 4
        self.lambdasquare *=  factor
        mrw, mrm = self.GenerateSfbm(size=size * M,subsample=subsample,t=t)
        self.lambdasquare /= factor
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

    #@njit
    def Generate_sample(self,t,Delta,N_MC=100,size=4096,subsample=8, M=32,type_gen="ShiftedMRM"):
        sample = []
        if type_gen=="ShiftedMRM":
            for _ in range(N_MC):
                # plt.plot(np.log(self.GenerateSfbm(size=size * M, subsample=subsample)[1]))
                # plt.show()
                #
                # path = self.GenerateSfbm(size=size * M, subsample=subsample)[1][t:int(t + Delta) + 1]
                # print(path)
                # print('////////////////////////')
                # #dmm = np.diff(self.GenerateSfbm(size=size * M, subsample=subsample)[1][t:int(t + Delta) + 1])
                # dmm = np.diff(path)

                # dmm = np.append(dmm[0], dmm)
                #logvol = self.GeneratelogVol(size, subsample, M)[1]
                # plt.plot(logvol)
                # plt.show()
                #mrm =
                #sample.append(np.cumsum(dmm)[-1])
                #sample.append(np.cumsum(np.exp(self.GeneratelogVol(size, subsample, M)[1])[t:int(t + Delta) + 1])[-1])
                sample.append(np.cumsum(np.exp(self.GeneratelogVol(size, subsample, M,Delta)[1]))[-1])
        if type_gen=="MRW sample":
            for _ in range(size):
                sample.append(self.GenerateSfbm(size=size * M, subsample=subsample,t=t)[0][-1])
        return sample


class MultidimensionalSfbm:
    def __init__(self,Sfbm_models,brownian_correlations={},dimension = 1,H_list=[],lambdasquare_list=[],T_list=[],sigma_list=[]):
        if type(Sfbm_models)!=list:
            TypeError("MultidimensionalSfbm error: Sfbm_models is not of expected type, list")
        # if any(type(Sfbm_model)!=Sfbm for Sfbm_model in Sfbm_models):
        #     TypeError("MultidimensionalSfbm error: Sfbm_models is not of expected type, Sfbm")
        All_lists = [H_list,lambdasquare_list,T_list,sigma_list]
        if (bool(brownian_correlations)==True) and any(len(item)!=dimension for item in All_lists if item!=[]):
            TypeError("MultidimensionalSfbm error: all list's size should be equal to dimension")
        else:
            self.Sfbm_models = Sfbm_models
            self.brownian_correlations = brownian_correlations
            self.correlation_flag = False
            if Sfbm_models == [] or  bool(brownian_correlations) == True :
                self.dimension = dimension
                self.correlation_flag = True
                self.H_list = H_list
                self.lambdasquare_list = lambdasquare_list
                self.T_list = T_list
                self.sigma_list = sigma_list
            if Sfbm_models != []:
                self.dimension = len(Sfbm_models)
                self.H_list = [Sfbm_model.H for Sfbm_model in Sfbm_models]
                self.lambdasquare_list = [Sfbm_model.lambdasquare for Sfbm_model in Sfbm_models]
                self.T_list = [Sfbm_model.T for Sfbm_model in Sfbm_models]
                self.sigma_list = [Sfbm_model.sigma for Sfbm_model in Sfbm_models]
                if bool(brownian_correlations) == True:
                    self.correlation_flag = True
            self.T = np.mean(np.array(self.T_list))
            # else:
            #     self.correlation_flag = False
            #     self.dimension = len(Sfbm_models)
            #self.w_correlation_matrix = None   w will be iid


            # self.H_list = H_list
            # self.lambdasquare_list = lambdasquare_list
            # self.T_list = T_list

            # self.sigma_list = sigma_list


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

    def LogMRM_CovarianceMatrixBuilder_generalcase(self,size=4096, dt=.1,flag_with_simulation=False):
        timesteps = dt * np.arange(1, size)
        # big_cov_matrix=[[0 for _ in range(self.dimension*(size-1))] for _ in range(self.dimension*(size-1))]
        covariance_random_field = lambda i,j,t,s:self.lambdasquare_list[i]*self.lambdasquare_list[j]/((self.H_list[i]+self.H_list[j])*(1-(self.H_list[i]+self.H_list[j])))*(self.T**(self.H_list[i]+self.H_list[j])-abs(t-s)**(self.H_list[i]+self.H_list[j]))
        # for i in range(self.dimension*(size-1)):
        #     for j in range(self.dimension*(size-1)):
        #         hurst_i,hurst_j = i//(size-1),j//(size-1)
        #         timestep_i,timestep_j = i%(size-1),j%(size-1)
        #         # print('hurst_i,hurst_j,timestep_i,timestep_j = ',hurst_i,hurst_j,timestep_i,timestep_j)
        #         big_cov_matrix[i][j] = covariance_random_field(hurst_i,hurst_j,timesteps[timestep_i], timesteps[timestep_j])
        big_cov_matrix = [[covariance_random_field(i//(size-1),j//(size-1),timesteps[i%(size-1)], timesteps[j%(size-1)]) for j in range(self.dimension * (size - 1))] for i in range(self.dimension * (size - 1))]
        mean_vector = np.array([self.lambdasquare_list[i//(size-1)] * self.T ** (
                    2 * self.H_list[i//(size-1)]) / (4 * (
                    self.H_list[i//(size-1)] * (1 - 2 * self.H_list[i//(size-1)]))) for i in
                                range(self.dimension * (size - 1))])
        if flag_with_simulation==False:
            return mean_vector,big_cov_matrix
        else:
            gaussian_instance_process = GaussianProcess('LastWaveFFT', big_cov_matrix)
            big_random_vector = gaussian_instance_process.Generate(1)[0]
            multipleLogMRM_paths = np.array_split(big_random_vector,self.dimension)
            return multipleLogMRM_paths

    def GenerateMultidimensionalSfbm(self,size=4096, subsample=8,brownian_correl_method = 'Brownian correlates - classical',generation_type = "Indep log mrm"):
        log_components = []
        if (self.correlation_flag == False) and (generation_type == "Indep log mrm"):
            if self.Sfbm_models == []:
                ValueError("MultidimensionalSfbm error: self.Sfbm_models should not be empty if correlation_flag =False")
            else:
                for Sfbm_model in self.Sfbm_models:
                    log_components.append(Sfbm_model.GenerateSfbm(size, subsample))
        else:
            log_mrm_matrix, dt = [], 1 / subsample
            if generation_type == "Indep log mrm":
                for dimension in range(self.dimension):
                    Sfbm_model = Sfbm(self.H_list[dimension], self.lambdasquare_list[dimension], self.T_list[dimension],self.sigma_list[dimension])
                    m, corr = Sfbm_model.SfbmCorrelation(size=(size - 1) * subsample, dt=dt)
                    log_mrm = GaussianProcess('LastWaveFFT', corr).Generate(size * subsample)
                    log_mrm = log_mrm + m
                    log_mrm = log_mrm[::subsample]
                    log_mrm_matrix.append(log_mrm + m)
            if generation_type == "Non Indep log mrm":
                # log_mrm_matrix = self.LogMRM_CovarianceMatrixBuilder_generalcase((size - 1) * subsample,dt,flag_with_simulation=True)
                
                # THIS
                # log_mrm_matrix = self.LogMRM_CovarianceMatrixBuilder_generalcase((size) * subsample+1, dt,flag_with_simulation=True)

                # print("log_mrm_matrix = ",np.array(log_mrm_matrix),len(log_mrm_matrix[0]),(size) * subsample,(size) , subsample)
                # print("np.hsplit(log_mrm_matrix,size) = ",len(np.hsplit(np.array(log_mrm_matrix),subsample)[0][0]))
                # log_mrm_matrix = [log_mrm[::subsample] for log_mrm in log_mrm_matrix]

                # THIS
                # log_mrm_matrix = list(np.hsplit(np.array(log_mrm_matrix), subsample)[0])

                log_mrm_matrix = list(np.hsplit(np.array(self.LogMRM_CovarianceMatrixBuilder_generalcase((size) * subsample+1, dt,flag_with_simulation=True)), subsample)[0])

            brownian_correlation_matrix = np.array([])
            if brownian_correl_method == 'Brownian correlates - classical':
                brownian_correlation_matrix = self.CorrelationMatrixBuilder_from_correlations(
                    self.brownian_correlations)
            if brownian_correl_method == 'Brownian correlates - random correl matrix':
                eigen_vectors = ConstructBasis(self.dimension)  # canonical basis
                eigen_values = np.exp(log_mrm_matrix[0])  # w are independant
                brownian_correlation_matrix = SumMultipleArrays( [eigen_values[i] * np.outer(eigen_vectors[i], eigen_vectors[i]) for i in range(self.dimension)])
            brownian_correlation_matrix_fact = np.linalg.cholesky(brownian_correlation_matrix)
            gg = np.random.normal(0, 1, (self.dimension, size))
            brownian_increments = np.array([brownian_correlation_matrix_fact @ gg[:, j] for j in range(size)]).T

            # print("log_mrm_matrix = ",log_mrm_matrix)
            # print("brownian_increments = ", brownian_increments)
            # print("self.sigma_list =",self.sigma_list)
            log_components = [(np.cumsum(np.exp(w) * brownian_increment * sigma * np.sqrt(dt)), np.array([])) for w, brownian_increment, sigma in  zip(log_mrm_matrix, brownian_increments, self.sigma_list)]
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


    def GeneratelogVolMultidimSfbm_Index(self,weights,method='quadratic variation estimate',size=4096, building_type = 'mrw',subsample=8,M=32,brownian_correl_method = 'Brownian correlates - classical',generation_type = "Indep log mrm"):  #,sigma=1, M=32
        # factor = 1
        # if self.H > 0:
        #     factor = M ** (-2 * self.H) / 4
        # self.lambdasquareintermittency *=  factor
        # print("INSIDE")
        trajectories = self.GenerateMultidimensionalSfbm(size*M, subsample,brownian_correl_method,generation_type)
        # print("trajectories = ",trajectories)

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

    def ConstructIndicestrajectories(self,size,subsample=4,building_type ='mrm and mrw',brownian_correl_method = 'Brownian correlates - classical',generation_type = "Indep log mrm"):
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
            Sfbms_generation_example = Index.GenerateMultidimensionalSfbm(size, subsample,brownian_correl_method,generation_type)

            indices_trajectories = Index.Index_Builder(weights, Sfbms_generation_example,building_type)
            Indicestrajectories.append(indices_trajectories)

            list_index+=1
        return Indicestrajectories


    def ConstructLogVolIndicestrajectories(self,size,subsample=8,method='quadratic variation estimate',keys = [],M = 32,brownian_correl_method = 'Brownian correlates - classical',building_type = 'mrw',generation_type = "Indep log mrm"):

        log_vol_indices_trajectories,indices_trajectories = [],self.ConstructIndicestrajectories(size*M,subsample,building_type,brownian_correl_method,generation_type)
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
