#  Paper:   "From Rough to Multifractal volatility: the log S-fBM model"
#  Purpose: Testing

# Author : Jean-Francois MUZY, Othmane ZARHALI


# Importations
from LogSfbmModel import Sfbm

import numpy as np
import matplotlib.pyplot as plt
from math import log,exp,sqrt
from scipy import optimize, special,interpolate

from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")

class VarIndexHurst:
    def __init__(self,brownian_correlations={},H_list=[],alpha_list=[],lambdasquare_list=[],T_list=[],sigma_list=[]):
        if type(brownian_correlations) != set:
            TypeError("VarIndexHurst error: H_list is not of expected type, list")
        All_lists = [H_list, alpha_list,lambdasquare_list, T_list, sigma_list]
        dimension = len(H_list)
        #print("All_lists = ",All_lists)
        if any(len(item) != dimension for item in All_lists) :
            TypeError("VarIndexHurst error: all list's size should be equal to dimension")
        else:
            self.dimension = dimension
            self.brownian_correlations = brownian_correlations
            self.H_list = H_list
            self.alpha_list = alpha_list
            self.lambdasquare_list = lambdasquare_list
            self.T_list = T_list
            self.sigma_list = sigma_list
            self.nus_square_list = np.array([lambdasquare/(H*(1-2*H)) for (lambdasquare,H) in zip(self.lambdasquare_list,self.H_list)])
            self.nu_square = np.mean(self.nus_square_list)
            self.T = np.mean(np.array(self.T_list))
            self.nu_inf_square = min(self.nus_square_list)
            self.nu_sup_square = max(self.nus_square_list)

    def ComputeHurst(self,brownian_correl_method = 'Brownian correlates - classical',g_i_j_matrix = None):
        if brownian_correl_method == 'Brownian correlates - classical':
            if g_i_j_matrix != None:
                ValueError("VarIndexHurst ComputeHurst error: g_i_j_matrix should be None")
            else:
                S = 0
                for i in range(self.dimension):
                    for j in range(self.dimension):
                        if i == j:
                            S += self.alpha_list[i] * self.alpha_list[j] * exp(
                                1 / 2 * (self.nus_square_list[i] * self.T ** (2 * self.H_list[i]) +
                                         self.nus_square_list[
                                             j] * self.T ** (2 * self.H_list[i])))
                        elif i < j:
                            S += self.alpha_list[i] * self.alpha_list[j] * self.brownian_correlations[(i, j)] * exp(
                                1 / 2 * (self.nus_square_list[i] * self.T ** (2 * self.H_list[i]) +
                                         self.nus_square_list[j] * self.T ** (2 * self.H_list[i])))
                        else:
                            S += self.alpha_list[i] * self.alpha_list[j] * self.brownian_correlations[(j, i)] * exp(
                                1 / 2 * (self.nus_square_list[i] * self.T ** (2 * self.H_list[i]) +
                                         self.nus_square_list[j] * self.T ** (
                                                 2 * self.H_list[i])))
                # print("(2/3*self.nu_square) = ",(2/3*self.nu_square))
                # print("log(S) = ",log(S))
                # print("log((2/3*self.nu_square)*log(S)) = ", log((2/3*self.nu_square)*log(S)))
                return log((2 / 3 * self.nu_square) * log(S)) / log(self.T ** 2)
        if brownian_correl_method == 'General case':
            if g_i_j_matrix == None:
                ValueError("VarIndexHurst ComputeHurst error: g_i_j_matrix should not be None")
            else:
                I = lambda T, i, j: sqrt(self.lambdasquare_list[i]) * sqrt(self.lambdasquare_list[j]) * self.T ** (
                            self.H_list[i] + self.H_list[j]) * (self.H_list[i] + self.H_list[j] + 1) / (
                                                (self.H_list[i] + self.H_list[j]) * (
                                                    1 - (self.H_list[i] + self.H_list[j])))
                S = 0
                for i in range(self.dimension):
                    for j in range(self.dimension):
                        if i == j:
                            S += self.alpha_list[i] * self.alpha_list[j] * exp(
                                1 / 2 * (self.nus_square_list[i] * self.T ** (2 * self.H_list[i]) +
                                         self.nus_square_list[
                                             j] * self.T ** (2 * self.H_list[i]))) * exp(I(self.T, i, j))
                        elif i < j:
                            S += self.alpha_list[i] * self.alpha_list[j] * self.brownian_correlations[(i, j)] * exp(
                                1 / 2 * (self.nus_square_list[i] * self.T ** (2 * self.H_list[i]) +
                                         self.nus_square_list[j] * self.T ** (2 * self.H_list[i]))) * exp(
                                g_i_j_matrix[(i, j)] * I(self.T, i, j))
                        else:
                            S += self.alpha_list[i] * self.alpha_list[j] * self.brownian_correlations[(j, i)] * exp(
                                1 / 2 * (self.nus_square_list[i] * self.T ** (2 * self.H_list[i]) +
                                         self.nus_square_list[j] * self.T ** (
                                                 2 * self.H_list[i]))) * exp(g_i_j_matrix[(j, i)] * I(self.T, j, i))
                # print("(2/3*self.nu_square) = ", (2 / 3 * self.nu_square))
                # print("log(S) = ", log(S))
                # print("log((2/3*self.nu_square)*log(S)) = ", log((2 / 3 * self.nu_square) * log(S)))
                return log((2 / 3 * self.nu_square) * log(S)) / log(self.T ** 2)


    def ComputeBounds(self,brownian_correl_method = 'Brownian correlates - classical'):
        if brownian_correl_method == 'Brownian correlates - classical':
            # Check sign rho_i,j>0
            check_positivity = any(self.brownian_correlations[(i, j)] < 0 for i in range(1,self.dimension) for j in range(i+1, self.dimension))
            # Check sign 0<rho_i,j<1
            check_boundedness = any(self.brownian_correlations[(i, j)] < 0 or self.brownian_correlations[(i, j)]>1 for i in range(1,self.dimension) for j in range(i+1, self.dimension))
            A_d = 0
            for i in range(self.dimension):
                for j in range(self.dimension):
                    if i == j:
                        A_d += self.alpha_list[i] * self.alpha_list[j]
                    elif i < j:
                        A_d += self.alpha_list[i] * self.alpha_list[j] * self.brownian_correlations[(i, j)]
                    else:
                        A_d += self.alpha_list[i] * self.alpha_list[j] * self.brownian_correlations[(j, i)]
            if check_positivity == False:
                print("log(A_d) = ", log(A_d))
                print("A_d = ", A_d)
                return {'Lowerbound': (1 / log(self.T ** 2)) * log(
                    2 / (3 * self.nu_inf_square) * log(A_d) + (2 * self.nu_inf_square) / (3 * self.nu_sup_square)),
                        'Upperbound': (1 / log(self.T ** 2)) * log(
                            2 / (3 * self.nu_inf_square) * log(A_d) + (2 * self.nu_inf_square) / (3 * self.nu_sup_square) * self.T)}
            if check_boundedness == False:
                return {'Lowerbound': (1 / log(self.T ** 2)) * log(
                    2 / (3 *  self.nu_inf_square) * log(A_d) + (2 * self.nu_inf_square) / (3 * self.nu_sup_square)),
                        'Upperbound': (1 / log(self.T ** 2)) * log((2 * self.nu_inf_square) / (3 * self.nu_sup_square))+1/2}
        else:
            return "Not available"

    def ComputeLinearLowerbound(self,brownian_correl_method = 'Brownian correlates - classical'):
        if brownian_correl_method == 'Brownian correlates - classical':
            A_d = 0
            for i in range(self.dimension):
                for j in range(self.dimension):
                    if i == j:
                        A_d += self.alpha_list[i] * self.alpha_list[j]
                    elif i <= j:
                        A_d += self.alpha_list[i] * self.alpha_list[j] * self.brownian_correlations[(i, j)]
                    else:
                        A_d += self.alpha_list[i] * self.alpha_list[j] * self.brownian_correlations[(j, i)]
            S=0
            for i in range(self.dimension):
                for j in range(self.dimension):
                    if i == j:
                        S += self.alpha_list[i] * self.alpha_list[j]/ A_d * (self.T ** 2 / 2) * self.nu_inf_square * (self.H_list[i] + self.H_list[j])
                    elif i < j:
                        S += self.alpha_list[i] * self.alpha_list[j] * self.brownian_correlations[(i, j)]/A_d*(self.T**2/2)*self.nu_inf_square*(self.H_list[i]+self.H_list[j])
                    else:
                        S += self.alpha_list[i] * self.alpha_list[j] * self.brownian_correlations[(j, i)]/A_d*(self.T**2/2)*self.nu_inf_square*(self.H_list[i]+self.H_list[j])
            if self.T>exp(-0.5):
                # H check in ]1/log(T^2),1/2[
                check_boundedness_Hs = any((self.H_list[i] < 1 / log(self.T ** 2) or self.H_list[i] > 0.5) for i in range(self.dimension))
                check_positivity = any(self.brownian_correlations[(i, j)] < 0 for i in range(1, self.dimension) for j in  range(i + 1, self.dimension))
                print("im here linear lb, check_boundedness_Hs  = ", check_boundedness_Hs)
                if (check_boundedness_Hs == False) and (check_positivity == False):
                    if self.T<exp(0.5):
                        # Check sign rho_i,j>0
                       #print("im here linear lb, check_boundedness_Hs , check_positivity = ",check_boundedness_Hs,check_positivity)
                        if (check_boundedness_Hs == False) and (check_positivity == False):
                            if A_d>1:
                                return {'Lowerbound':2/(self.T**2*self.nu_sup_square)*(S-3/2*self.nu_sup_square) ,'Upperbound': None}
                    else:
                        #print("value = ", (2 / 3 * self.nu_sup_square) ,(A_d) , S ,self.T ** 2,log(self.T ** 2))
                        return {'Lowerbound': log((2 / 3 * self.nu_sup_square) * (log(A_d) + S)) / log(self.T ** 2),'Upperbound': None}
                else:
                    ValueError("VarIndexHurst ComputeLinearLowerbound error:  H_i's should be in ]1/log(T^2),1/2[ ")
        else:
            return "Not available"

    def ComputeFirstOrderApproximations(self,brownian_correl_method = 'Brownian correlates - classical',approximation_type= "T infty"):
        A_d = 0
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i == j:
                    A_d += self.alpha_list[i] * self.alpha_list[j]
                elif i < j:
                    A_d += self.alpha_list[i] * self.alpha_list[j] * self.brownian_correlations[(i, j)]
                else:
                    A_d += self.alpha_list[i] * self.alpha_list[j] * self.brownian_correlations[(j, i)]
        if brownian_correl_method == 'Brownian correlates - classical':
            if approximation_type == "T infty":
                return max(self.H_list)+log(2*self.nus_square_list[np.argmax(max(self.H_list))]/(3*self.nu_square))/log(self.T**2)
            if approximation_type == "Small intermittencies":
                double_sum=0
                for i in range(self.dimension):
                    for j in range(self.dimension):
                        if i == j:
                            #double_sum += 3*self.nu_square*self.alpha_list[i] * self.alpha_list[j]/(4*log(A_d)*A_d)*(self.nus_square_list[i]*self.T**(2*self.H_list[i])+self.nus_square_list[i]*self.T**(2*self.H_list[j]))
                            double_sum +=  self.nu_square * self.alpha_list[i] * self.alpha_list[j] / (
                                        2 * A_d) * (
                                                      self.nus_square_list[i] * self.T ** (2 * self.H_list[i]) +
                                                      self.nus_square_list[i] * self.T ** (2 * self.H_list[j]))

                        elif i < j:
                            #double_sum += 3*self.nu_square*self.alpha_list[i] * self.alpha_list[j] * self.brownian_correlations[(i, j)]/(4*log(A_d)*A_d)*(self.nus_square_list[i]*self.T**(2*self.H_list[i])+self.nus_square_list[i]*self.T**(2*self.H_list[j]))
                            double_sum +=  self.nu_square * self.alpha_list[i] * self.alpha_list[j] * \
                                          self.brownian_correlations[(i, j)] / (2 * A_d) * (
                                                      self.nus_square_list[i] * self.T ** (2 * self.H_list[i]) +
                                                      self.nus_square_list[i] * self.T ** (2 * self.H_list[j]))

                        else:
                            #double_sum +=  3*self.nu_square*self.alpha_list[i] * self.alpha_list[j] * self.brownian_correlations[(j, i)]/(4*log(A_d)*A_d)*(self.nus_square_list[i]*self.T**(2*self.H_list[i])+self.nus_square_list[i]*self.T**(2*self.H_list[j]))
                            double_sum +=  self.nu_square * self.alpha_list[i] * self.alpha_list[j] * \
                                          self.brownian_correlations[(j, i)] / (2 * A_d) * (
                                                      self.nus_square_list[i] * self.T ** (2 * self.H_list[i]) +
                                                      self.nus_square_list[i] * self.T ** (2 * self.H_list[j]))
                print("log(A_d) = ",log(A_d))
                print("A_d = ", A_d)
                print("log(A_d) + double_sum = ",log(A_d) + double_sum)
                print('double_sum = ',double_sum)
                #return(1/log(self.T**2))*(log(2/(3*self.nu_square)*log(A_d))+double_sum)
                return (1 / log(self.T ** 2)) * log(2 / (3 * self.nu_square) * (log(A_d) + double_sum))
        else:
            return "not available"

    def ComputeEvolution(self,evolution_type,with_asymptotics,hurst_arguments,brownian_correl_method = 'Brownian correlates - classical'):
        if type(hurst_arguments) != dict:
            TypeError("VarIndexHurst ComputeEvolution error: hurst_arguments is not of expected type, dict")
        else:
            length = len(hurst_arguments['T_lists'])
            if evolution_type == 'T without bounds':
                Ts = [np.mean(np.array(T_list)) for T_list in hurst_arguments['T_lists']]
                hursts = [VarIndexHurst(hurst_arguments['correl_lists'][i],hurst_arguments['H_lists'][i],hurst_arguments['alpha_lists'][i],hurst_arguments['lambda_square_lists'][i],hurst_arguments['T_lists'][i],hurst_arguments['sigma_lists'][i]).ComputeHurst(brownian_correl_method) for i in range(length)]
                print("hursts = ",hursts)
                if with_asymptotics == True:
                    asymptotic_T_hurst = [VarIndexHurst(hurst_arguments['correl_lists'][i],hurst_arguments['H_lists'][i],hurst_arguments['alpha_lists'][i],hurst_arguments['lambda_square_lists'][i],hurst_arguments['T_lists'][i],hurst_arguments['sigma_lists'][i]).ComputeFirstOrderApproximations(brownian_correl_method,"T infty") for i in range(length)]
                    plt.plot(Ts, hursts, label='Hurst index')
                    plt.plot(Ts, asymptotic_T_hurst, label=r'Asymptotic hurst when $T \rightarrow +\infty$')
                    plt.legend()
                    plt.title("Hurst index evolution with respect to T")
                    plt.show()
                else:
                    plt.plot(Ts, hursts, label='Hurst index')
                    plt.legend()
                    plt.title("Hurst index evolution with respect to T")
                    plt.show()
            if evolution_type == 'T with bounds':
                Ts = [np.mean(np.array(T_list)) for T_list in hurst_arguments['T_lists']]
                hursts_lowerbound = [VarIndexHurst(hurst_arguments['correl_lists'][i],hurst_arguments['H_lists'][i],hurst_arguments['alpha_lists'][i],hurst_arguments['lambda_square_lists'][i],hurst_arguments['T_lists'][i],hurst_arguments['sigma_lists'][i]).ComputeBounds(brownian_correl_method)['Lowerbound'] for i in range(length)]
                hursts_upperbound = [VarIndexHurst(hurst_arguments['correl_lists'][i],hurst_arguments['H_lists'][i],hurst_arguments['alpha_lists'][i],hurst_arguments['lambda_square_lists'][i],hurst_arguments['T_lists'][i],hurst_arguments['sigma_lists'][i]).ComputeBounds(brownian_correl_method)['Upperbound'] for i in range(length)]
                hursts = [VarIndexHurst(hurst_arguments['correl_lists'][i],hurst_arguments['H_lists'][i],hurst_arguments['alpha_lists'][i],hurst_arguments['lambda_square_lists'][i],hurst_arguments['T_lists'][i],hurst_arguments['sigma_lists'][i]).ComputeHurst(brownian_correl_method) for i in range(length)]
                plt.plot(Ts,hursts_lowerbound,label='Hurst index lowerbound')
                plt.plot(Ts, hursts_upperbound, label='Hurst index upperbound')
                plt.plot(Ts, hursts, label='Hurst index')
                plt.legend()
                plt.title("Hurst index evolution with bounds with respect to T")
                plt.show()
            if evolution_type == 'Intermittencies':
                lambdas = [np.sum(np.array(lambda_square_list)) for lambda_square_list in hurst_arguments['lambda_square_lists']]
                hursts = [VarIndexHurst(hurst_arguments['correl_lists'][i], hurst_arguments['H_lists'][i],
                                        hurst_arguments['alpha_lists'][i],
                                        hurst_arguments['lambda_square_lists'][i], hurst_arguments['T_lists'][i],
                                        hurst_arguments['sigma_lists'][i]).ComputeHurst(brownian_correl_method) for
                          i in range(length)]
                print("hursts = ", hursts)
                if with_asymptotics == True:
                    asymptotic_T_hurst = [
                        VarIndexHurst(hurst_arguments['correl_lists'][i], hurst_arguments['H_lists'][i],
                                      hurst_arguments['alpha_lists'][i], hurst_arguments['lambda_square_lists'][i],
                                      hurst_arguments['T_lists'][i],
                                      hurst_arguments['sigma_lists'][i]).ComputeFirstOrderApproximations(
                            brownian_correl_method, "Small intermittencies") for i in range(length)]
                    # plt.plot(lambdas, np.exp(log(self.T**2)*np.array(hursts)), label='Hurst index')
                    # plt.plot(lambdas, np.exp(log(self.T**2)*np.array(asymptotic_T_hurst)), label=r'Asymptotic hurst when $\hat{\lambda} \rightarrow 0$')
                    plt.plot(lambdas, hursts, label='Hurst index')
                    plt.plot(lambdas, asymptotic_T_hurst,label=r'Asymptotic hurst when $\hat{\lambda} \rightarrow 0$')
                    plt.legend()
                    plt.title("Hurst index evolution with respect to intermittencies")
                    plt.show()
                else:
                    plt.plot(lambdas, hursts, label='Hurst index')
                    plt.legend()
                    plt.title("Hurst index evolution with respect to intermittencies")
                    plt.show()

    def ComputeCrossMRMCovCurvature(self,Delta,i,j,size=4096,subsample=8, M=32,h=1e-3):
        def kappa(Delta):
            #logSfbm_i, logSfbm_j =,
            # sample = logSfbm_i.GenerateSfbm(size=10)[1]
            # print("mean sample = ", np.mean(sample),sample,len(sample))

            shifted_MRM_sample_i_0 =  Sfbm(self.H_list[i], self.lambdasquare_list[i], self.T, self.sigma_list[i]).Generate_sample(0, Delta,size, size, subsample, M,'ShiftedMRM')
            shifted_MRM_sample_j_0 = Sfbm(self.H_list[j], self.lambdasquare_list[j], self.T, self.sigma_list[j]).Generate_sample(0, Delta,size, size, subsample, M,'ShiftedMRM')

            # objects = [logSfbm_i,logSfbm_j]
            # jobs_output = Parallel(n_jobs=2)(delayed(objects[i].Generate_sample)(0, Delta,size, size, subsample, M,'ShiftedMRM') for i in range(2))
            # shifted_MRM_sample_i_0,shifted_MRM_sample_j_0 = jobs_output[0],jobs_output[1]

            #print("jobs done ")
            # print("self.T = ",self.T)
            # print("mean d = ", np.mean(shifted_MRM_sample_i_0),shifted_MRM_sample_i_0,len(shifted_MRM_sample_i_0))
            # print('***********************')
            # print("len(shifted_MRM_sample_i_0) = ",len(shifted_MRM_sample_i_0))
            print("check stdev=  ",np.mean(shifted_MRM_sample_i_0) , 1/sqrt(size)*np.std(shifted_MRM_sample_i_0),"!!",np.mean(shifted_MRM_sample_j_0),1/sqrt(size)*np.std(shifted_MRM_sample_j_0))
            print("cov check =  ",np.cov(shifted_MRM_sample_i_0, shifted_MRM_sample_j_0))
            print('*****************************************')


            print("check cov = ",np.mean((shifted_MRM_sample_i_0-np.mean(shifted_MRM_sample_i_0))*(shifted_MRM_sample_j_0-np.mean(shifted_MRM_sample_j_0))))
            print("check std err= ",1/sqrt(size)*np.std((shifted_MRM_sample_i_0-np.mean(shifted_MRM_sample_i_0))*(shifted_MRM_sample_j_0-np.mean(shifted_MRM_sample_j_0))))
            print('*****************************************')
            # return np.mean((shifted_MRM_sample_i_0-np.mean(shifted_MRM_sample_i_0))*(shifted_MRM_sample_j_0-np.mean(shifted_MRM_sample_j_0)))
            return np.cov(shifted_MRM_sample_i_0, shifted_MRM_sample_j_0)[0,1]


        #kappas = [kappa(Delta) for Delta in np.linspace(2, 100, 100)]
        # import matplotlib.pyplot as plt
        # plt.plot(kappas)
        # plt.title("kappa")
        # plt.show()

        #kappa_Delta_plus_h,kappa_Delta_minus_h,kappa_Delta = kappa(Delta+h) ,kappa(Delta-h) ,kappa(Delta)
        #print("kappa(Delta+h) = ",(kappa_Delta_plus_h+kappa_Delta_minus_h-2*kappa_Delta),kappa_Delta_plus_h,kappa_Delta_minus_h,2*kappa_Delta)
        #curvature_ij = (kappa(Delta+h)+kappa(Delta-h)-2*kappa(Delta))/h**2
        #print("check curvature (kappa_Delta_plus_h+kappa_Delta_minus_h-2*kappa_Delta) = ",(kappa(Delta+h)+kappa(Delta-h)-2*kappa(Delta)))
        return (kappa(Delta+h)+kappa(Delta-h)-2*kappa(Delta))/h**2

    def ComputeSmoothCrossCov(self,delta,i,j,size=4096,type_cov="MRMCovCurvature_normalized",subsample=8, M=32,h=1e-7):
        if type_cov=="MRMCovCurvature_normalized":
            Delta_range = np.linspace(0,2,500)

            #curvatures = [self.ComputeCrossMRMCovCurvature(Delta, i, j, size, subsample, M, h) for Delta in Delta_range]
            curvatures = Parallel(n_jobs=60)(delayed(self.ComputeCrossMRMCovCurvature)(Delta, i, j, size, subsample, M, h) for Delta in Delta_range)
            # import matplotlib.pyplot as plt
            # plt.plot(curvatures)
            # plt.show()
            # print("curvatures = ", curvatures)
            # coefficients = np.polynomial.hermite.hermfit(Delta_range, curvatures, 3)
            # return np.sum([coefficients[i] * special.hermite(i, monic=True)(delta) for i in range(len(coefficients))])
            tck = interpolate.splrep(Delta_range, curvatures)


            # import matplotlib.pyplot as plt
            # import QuantLib as ql
            # i = ql.CubicNaturalSpline(list(Delta_range), list(curvatures))
            # plt.plot([interpolate.splev(delta, tck) for delta in np.linspace(0,1,1000)],color='red')
            # #plt.plot([i(x) for x in np.linspace(0,2,1000)],color='blue')
            # plt.title("check interpolation quantlib vs splev")
            # plt.show()
            #print("interpolate.splev(delta, tck) = ",interpolate.splev(delta, tck),(self.nus_square_list[i]*self.T**(2*self.H_list[i])+self.nus_square_list[j]*self.T**(2*self.H_list[j])))
            #return interpolate.CubicHermiteSpline(delta,Delta_range, curvatures)#*exp(-0.5*(self.nus_square_list[i]*self.T**(2*self.H_list[i])+self.nus_square_list[j]*self.T**(2*self.H_list[j])))
            return interpolate.splev(delta, tck)*exp(-0.5*(self.nus_square_list[i]*self.T**(2*self.H_list[i])+self.nus_square_list[j]*self.T**(2*self.H_list[j])))
        if type_cov=="MRWCov":
            def covMRW(t):
                logSfbm_i, logSfbm_j = Sfbm(self.H_list[i], self.lambdasquare_list[i], self.T,
                                            self.sigma_list[i]), Sfbm(
                    self.H_list[j], self.lambdasquare_list[j], self.T, self.sigma_list[j])
                shifted_MRW_sample_i_t = logSfbm_i.Generate_sample(t, 0,size*M, size, subsample, M, 'MRW sample')
                shifted_MRW_sample_j_t = logSfbm_j.Generate_sample(t, 0,size*M, size, subsample, M, 'MRW sample')
                return np.cov(shifted_MRW_sample_i_t, shifted_MRW_sample_j_t)[0, 1]
            #covs = [covMRW(t) for t in range(0,500,20)]
            #import matplotlib.pyplot as plt
            # plt.plot(covs)
            # plt.show()
            # print("covs = ",covs)
            return covMRW(delta)


    def g_i_j_Calibration(self,Delta,i,j,size=4096,subsample=8, M=32,h=1e-3,method = 'root finding'):
        if method == 'root finding':
            # I_T00 = sqrt(self.lambdasquare_list[i] * self.lambdasquare_list[j]) * self.T ** (
            #         self.H_list[i] + self.H_list[j]) * (self.H_list[i] + self.H_list[j] + 1) / (
            #                 (self.H_list[i] + self.H_list[j]) * (
            #                 1 - (self.H_list[i] + self.H_list[j])))
            # I_T0Delta = sqrt(self.lambdasquare_list[i] * self.lambdasquare_list[j]) * ((self.T ** (
            #             self.H_list[i] + self.H_list[j] - 1) * (self.T - Delta) - Delta ** (self.H_list[i] +
            #                                                                                 self.H_list[j])) / (
            #                                                                                    (self.H_list[i] +
            #                                                                                     self.H_list[j]) * (
            #                                                                                            1 - (self.H_list[
            #                                                                                                     i] +
            #                                                                                                 self.H_list[
            #                                                                                                     j]))) - (
            #                                                                                        (
            #                                                                                                    self.T - Delta) * self.T ** (
            #                                                                                                self.H_list[
            #                                                                                                    i] +
            #                                                                                                self.H_list[
            #                                                                                                    j] - 1)) / (
            #                                                                                        self.H_list[i] +
            #                                                                                        self.H_list[j] - 1))
            # der_xI_T00 = sqrt(self.lambdasquare_list[i]) * sqrt(self.lambdasquare_list[j]) * self.T ** (
            #         self.H_list[i] + self.H_list[j] - 1) * (self.H_list[i] + self.H_list[j] + 1) / (
            #                      (self.H_list[i] + self.H_list[j]) * (
            #                      1 - (self.H_list[i] + self.H_list[j])))
            # der_yI_T0Delta = sqrt(self.lambdasquare_list[i] * self.lambdasquare_list[j]) * (
            #             (Delta ** (self.H_list[i] + self.H_list[j] - 1)) / (
            #                 1 - (self.H_list[i] + self.H_list[j])) - self.T ** (self.H_list[i] + self.H_list[j] - 1) / (
            #                         self.H_list[i] + self.H_list[j] - 1))
            # print("check g_ij= ",-self.ComputeSmoothCrossCov(Delta,i,j,size,"MRMCovCurvature_normalized",subsample, M,h)/2)
            # equation = lambda x: abs(x * (Delta * (der_yI_T0Delta * exp(x * I_T0Delta) - der_xI_T00 * exp(
            #     x * I_T00)) - 2) - self.ComputeSmoothCrossCov(Delta,i,j,size,"MRMCovCurvature_normalized",subsample, M,h))**2
            # print(equation(0))
            # print("bef newton = ")
            #return ('g_ij,minumum,maxiter reached, num function call)=',optimize.brent(equation, maxiter=15,full_output=True))
            return -self.ComputeSmoothCrossCov(Delta,i,j,size,"MRMCovCurvature_normalized",subsample, M,h)/2
        else:
            pass


    def g_i_j_MatrixCalibration(self,Delta):
        g_ij_matrix = {}
        for i in range(self.dimension):
            for j in range(i,self.dimension):
                g_ij_matrix[(i,j)]=self.g_i_j_Calibration(Delta,i,j)
        return g_ij_matrix



