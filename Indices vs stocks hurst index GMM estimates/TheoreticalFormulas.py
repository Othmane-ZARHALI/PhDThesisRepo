#  Paper:   "From Rough to Multifractal volatility: the log S-fBM model"
#  Purpose: Testing

# Author : Jean-Francois MUZY, Othmane ZARHALI


# Importations
from LogSfbmModel import Sfbm

import numpy as np
import matplotlib.pyplot as plt
from math import log,exp,sqrt,pi
from scipy import optimize, special,interpolate
# from sklearn.linear_model import LinearRegression

from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")

class VarIndexHurst:
    def __init__(self,brownian_correlations={},H_list=[],alpha_list=[],lambdasquare_list=[],T_list=[],sigma_list=[]):
        if type(brownian_correlations) != set:
            TypeError("VarIndexHurst error: H_list is not of expected type, list")
        All_lists = [H_list, alpha_list,lambdasquare_list, T_list, sigma_list]
        dimension = len(H_list)
        if dimension < 2:
            ValueError("VarIndexHurst error: dimension should be grater or equal than 2")
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
            return log((2 / 3 * self.nu_square) * log(S)) / log(self.T ** 2)

        if brownian_correl_method == 'General case':
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
                                     self.nus_square_list[j] * self.T ** (2 * self.H_list[i]))) * exp(I(self.T, i, j))
                    else:
                        S += self.alpha_list[i] * self.alpha_list[j] * self.brownian_correlations[(j, i)] * exp(
                            1 / 2 * (self.nus_square_list[i] * self.T ** (2 * self.H_list[i]) +
                                     self.nus_square_list[j] * self.T ** (
                                             2 * self.H_list[i]))) * exp(I(self.T, j, i))
            return log((2 / 3 * self.nu_square) * log(S)) / log(self.T ** 2)


    def ComputeHurst_log_small_intermittencies(self,method = 'linreglog moments'):
        def logMRM_Moments(q,tau,Delta):
            d=self.dimension
            gamma_2 = lambda i, j: self.T ** (self.H_list[i] + self.H_list[j] - 1) / ((self.H_list[i] + self.H_list[j] - 1) * (self.H_list[i] + self.H_list[j]))
            if d % 2 == 0:
                sum_term = np.sum(np.array([2 * (self.alpha_list[i] ** 2) * (self.alpha_list[j] ** 2) * sqrt(
                    self.lambdasquare_list[i]) * sqrt(self.lambdasquare_list[j]) * (d - 1) * (
                                                    2 * (
                                                    gamma_2(i, j) / (self.H_list[i] + self.H_list[j] + 1) * tau ** (
                                                    self.H_list[i] + self.H_list[j] + 1) / Delta)) for
                                            (i, j) in zip(range(d), range(d))]))
            else:
                sum_term = 1*np.sum(np.array([2 * (self.alpha_list[i] ** 2) * (self.alpha_list[j] ** 2) * sqrt(
                    self.lambdasquare_list[i]) * sqrt(self.lambdasquare_list[j]) * (d - 2)* (2 * (
                                                    gamma_2(i, j) / (self.H_list[i] + self.H_list[j] + 1) * tau ** (
                                                    self.H_list[i] + self.H_list[j] + 1) / Delta)) for (i, j)
                                            in zip(range(d), range(d))]))
            return log(2 ** (q / 2) * special.gamma((q + 1) / 2) * (-sum_term)**(q / 2) / sqrt(pi))
        q = 3
        Delta = 1e-0
        if method == 'linreglog moments':
            logtauoverT_range = np.linspace(-100, 0, 100000)
            tau_range_fromrange = self.T * np.exp(logtauoverT_range)
            objective_function_values = [logMRM_Moments(q, tau, Delta) for tau in tau_range_fromrange]
            hurst_index_as_slope = np.polyfit(logtauoverT_range, objective_function_values, 1)[0]
            return hurst_index_as_slope / q
        if method == 'root finding':
            logtauoverT_range = np.linspace(0,5, 1000)
            tau_range_fromrange = self.T * np.exp(logtauoverT_range)
            def ObjectiveFunction(tau,H):
                z=Delta/tau
                factor = (2/3*H*(1-2*H)*log(sum(np.array(self.alpha_list)**2*np.exp(np.array(self.lambdasquare_list)*np.array(self.T_list)**(2*np.array(self.H_list))/(np.array(self.H_list)*(1-2*np.array(self.H_list)))))))
                g = (- 2 * abs(z) ** (2 * H ) ) / (  H * (1 - 2 * H) * (2 * H + 1) * (2 * H + 2))
                return (exp(logMRM_Moments(q,tau,Delta))-(2 ** (q / 2) / sqrt(pi) * special.gamma((q + 1) / 2)*(factor*g) ** (q / 2) * (tau/self.T) ** (q *H)))
            objective_function = lambda H:np.mean([ObjectiveFunction(tau,H) for tau in tau_range_fromrange])
            newton_sol = optimize.newton(objective_function, x0=np.array(0.01),tol=1.48e-18,maxiter=100)  # bounds=((1e-10), (0.45))
            return newton_sol



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
                        if (check_boundedness_Hs == False) and (check_positivity == False):
                            if A_d>1:
                                return {'Lowerbound':2/(self.T**2*self.nu_sup_square)*(S-3/2*self.nu_sup_square) ,'Upperbound': None}
                    else:
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
                            double_sum +=  self.nu_square * self.alpha_list[i] * self.alpha_list[j] / (
                                        2 * A_d) * (
                                                      self.nus_square_list[i] * self.T ** (2 * self.H_list[i]) +
                                                      self.nus_square_list[i] * self.T ** (2 * self.H_list[j]))

                        elif i < j:
                            double_sum +=  self.nu_square * self.alpha_list[i] * self.alpha_list[j] * \
                                          self.brownian_correlations[(i, j)] / (2 * A_d) * (
                                                      self.nus_square_list[i] * self.T ** (2 * self.H_list[i]) +
                                                      self.nus_square_list[i] * self.T ** (2 * self.H_list[j]))

                        else:
                            double_sum +=  self.nu_square * self.alpha_list[i] * self.alpha_list[j] * \
                                          self.brownian_correlations[(j, i)] / (2 * A_d) * (
                                                      self.nus_square_list[i] * self.T ** (2 * self.H_list[i]) +
                                                      self.nus_square_list[i] * self.T ** (2 * self.H_list[j]))
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
                if with_asymptotics == True:
                    asymptotic_T_hurst = [
                        VarIndexHurst(hurst_arguments['correl_lists'][i], hurst_arguments['H_lists'][i],
                                      hurst_arguments['alpha_lists'][i], hurst_arguments['lambda_square_lists'][i],
                                      hurst_arguments['T_lists'][i],
                                      hurst_arguments['sigma_lists'][i]).ComputeFirstOrderApproximations(
                            brownian_correl_method, "Small intermittencies") for i in range(length)]
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






