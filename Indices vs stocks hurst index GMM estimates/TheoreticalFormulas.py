#  Paper:   "From Rough to Multifractal volatility: the log S-fBM model"
#  Purpose: Testing

# Author : Jean-Francois MUZY, Othmane ZARHALI


# Importations
import numpy as np
import matplotlib.pyplot as plt
from math import log,exp

import warnings
warnings.filterwarnings("ignore")

class VarIndexHurst:
    def __init__(self,brownian_correlations={},H_list=[],alpha_list=[],lambdasquare_list=[],T_list=[],sigma_list=[]):
        if type(brownian_correlations) != set:
            TypeError("VarIndexHurst error: H_list is not of expected type, list")
        All_lists = [H_list, alpha_list,lambdasquare_list, T_list, sigma_list]
        dimension = len(H_list)
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

    def ComputeHurst(self,brownian_correl_method = 'Brownian correlates - classical'):
        if brownian_correl_method == 'Brownian correlates - classical':
            S=0
            for i in range(self.dimension):
                for j in range(self.dimension):
                    if i == j:
                        S += self.alpha_list[i] * self.alpha_list[j] * exp(
                            1 / 2 * (self.nus_square_list[i] * self.T ** (2 * self.H_list[i]) + self.nus_square_list[
                                j] * self.T ** (2 * self.H_list[i])))
                    elif i < j:
                        S+=self.alpha_list[i]*self.alpha_list[j]*self.brownian_correlations[(i,j)]*exp(1/2*(self.nus_square_list[i]*self.T**(2*self.H_list[i])+self.nus_square_list[j]*self.T**(2*self.H_list[i])))
                    else:
                        S += self.alpha_list[i] * self.alpha_list[j] * self.brownian_correlations[(j,i)] * exp(
                            1 / 2 * (self.nus_square_list[i] * self.T ** (2 * self.H_list[i]) + self.nus_square_list[j] * self.T ** (
                                    2 * self.H_list[i])))
            # print("(2/3*self.nu_square) = ",(2/3*self.nu_square))
            # print("log(S) = ",log(S))
            # print("log((2/3*self.nu_square)*log(S)) = ", log((2/3*self.nu_square)*log(S)))
            return log((2/3*self.nu_square)*log(S))/log(self.T**2)
        else:
            return "Not available"

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
                            double_sum += 3*self.nu_square*self.alpha_list[i] * self.alpha_list[j]/(4*log(A_d)*A_d)*(self.nus_square_list[i]*self.T**(2*self.H_list[i])+self.nus_square_list[i]*self.T**(2*self.H_list[j]))

                        elif i < j:
                            double_sum += 3*self.nu_square*self.alpha_list[i] * self.alpha_list[j] * self.brownian_correlations[(i, j)]/(4*log(A_d)*A_d)*(self.nus_square_list[i]*self.T**(2*self.H_list[i])+self.nus_square_list[i]*self.T**(2*self.H_list[j]))
                        else:
                            double_sum +=  3*self.nu_square*self.alpha_list[i] * self.alpha_list[j] * self.brownian_correlations[(j, i)]/(4*log(A_d)*A_d)*(self.nus_square_list[i]*self.T**(2*self.H_list[i])+self.nus_square_list[i]*self.T**(2*self.H_list[j]))
                print("log(A_d) = ",log(A_d))
                return(1/log(self.T**2))*(log(2/(3*self.nu_square)*log(A_d))+double_sum)

    def ComputeEvolution(self,type,hurst_arguments,brownian_correl_method = 'Brownian correlates - classical'):
        print("hurst_arguments = ",hurst_arguments,type(hurst_arguments))
        if type(hurst_arguments) != dict:
            TypeError("VarIndexHurst ComputeEvolution error: hurst_arguments is not of expected type, dict")
        else:
            if type == 'T without bounds':
                length = len(hurst_arguments['T'])
                Ts = [np.mean(np.array(T_list)) for T_list in hurst_arguments['T']]
                hursts = [VarIndexHurst(hurst_arguments['correl'][i],hurst_arguments['H_list'][i],hurst_arguments['alpha_lits'][i],hurst_arguments['lambda_square_list'][i],hurst_arguments['T'][i],hurst_arguments['sigma'][i]).ComputeHurst(brownian_correl_method) for i in range(length)]
                plt.plot(Ts,hursts,label='Hurst index')
                plt.legend()
                plt.title("Hurst index evolution with respect to T")
                plt.show()
            if type == 'T with bounds':
                length = len(hurst_arguments['T'])
                Ts = [np.mean(np.array(T_list)) for T_list in hurst_arguments['T']]
                hursts_lowerbound = [VarIndexHurst(hurst_arguments['correl'][i],hurst_arguments['H_list'][i],hurst_arguments['alpha_lists'][i],hurst_arguments['lambda_square_list'][i],hurst_arguments['T'][i],hurst_arguments['sigma'][i]).ComputeBounds(brownian_correl_method)['Lowerbound'] for i in range(length)]
                hursts_upperbound = [VarIndexHurst(hurst_arguments['correl'][i], hurst_arguments['H_list'][i], hurst_arguments['alpha_lits'][i],hurst_arguments['lambda_square_list'][i], hurst_arguments['T'][i],hurst_arguments['sigma'][i]).ComputeBounds(brownian_correl_method)['Upperbound'] for i in range(length)]
                hursts = [VarIndexHurst(hurst_arguments['correl'][i], hurst_arguments['H_list'][i],hurst_arguments['alpha_lits'][i], hurst_arguments['lambda_square_list'][i],hurst_arguments['T'][i], hurst_arguments['sigma'][i]).ComputeHurst(brownian_correl_method) for i in range(length)]
                plt.plot(Ts,hursts_lowerbound,label='Hurst index lowerbound')
                plt.plot(Ts, hursts_upperbound, label='Hurst index upperbound')
                plt.plot(Ts, hursts, label='Hurst index upperbound')
                plt.legend()
                plt.title("Hurst index evolution with bounds with respect to T")
                plt.show()