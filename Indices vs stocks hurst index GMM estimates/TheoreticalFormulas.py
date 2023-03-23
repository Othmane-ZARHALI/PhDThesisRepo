#  Paper:   "From Rough to Multifractal volatility: the log S-fBM model"
#  Purpose: Testing

# Author : Jean-Francois MUZY, Othmane ZARHALI


# Importations
from LogSfbmModel import Sfbm
from GMMCalibration import *

import numpy as np
import matplotlib.pyplot as plt
from math import log,exp,sqrt,pi
from scipy import optimize, special,interpolate
from sklearn.linear_model import LinearRegression

from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=DeprecationWarning)

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
                        # S += self.alpha_list[i] * self.alpha_list[j] * exp(
                        #     1 / 2 * (self.nus_square_list[i] * self.T ** (2 * self.H_list[i]) +
                        #              self.nus_square_list[j] * self.T ** (2 * self.H_list[i])))
                        S += self.alpha_list[i]**2 * exp(
                            3 / 2 * (self.nus_square_list[i] * self.T ** (2 * self.H_list[i])))
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
                    self.H_list[i] + self.H_list[j]) / ((self.H_list[i] + self.H_list[j]) * (1 - (self.H_list[i] + self.H_list[j])))
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
            #gamma_2 = lambda i, j: self.T ** (self.H_list[i] + self.H_list[j] - 1) / ((self.H_list[i] + self.H_list[j] - 1) * (self.H_list[i] + self.H_list[j]))
            if d % 2 == 0:
                # sum_term = np.sum(np.array([2 * (self.alpha_list[i] ** 2) * (self.alpha_list[j] ** 2) * sqrt(
                #     self.lambdasquare_list[i]) * sqrt(self.lambdasquare_list[j]) * (d - 1) * (
                #                                     2 * (
                #                                     gamma_2(i, j) / (self.H_list[i] + self.H_list[j] + 1) * tau ** (
                #                                     self.H_list[i] + self.H_list[j] + 1) / Delta)) for
                #                             (i, j) in zip(range(d), range(d))]))
                # sum_term = np.sum(np.array([[(self.alpha_list[i] ** 2) * (self.alpha_list[j] ** 2) * sqrt(
                #     self.lambdasquare_list[i]) * sqrt(self.lambdasquare_list[j]) * (d - 1) * (
                #                                         (self.H_list[i] + self.H_list[j] + 2) / (
                #                                             1 - self.H_list[i] - self.H_list[j])
                #                                         * tau ** (self.H_list[i] + self.H_list[j]) * Delta ** 2) for i in range(d)] for j in range(d)]))
                # sum_term = np.sum(0.5*np.array([[(self.alpha_list[i] ** 2) * (self.alpha_list[j] ** 2) * sqrt(
                #     self.lambdasquare_list[i]) * sqrt(self.lambdasquare_list[j]) * (d - 1) * (1/ ( (self.H_list[i] + self.H_list[j])*(
                #                                      1 - self.H_list[i] - self.H_list[j]))
                #                                      * tau ** (self.H_list[i] + self.H_list[j]) * Delta ** 2) for i in
                #                              range(d)] for j in range(d)]))
                sum_term = np.sum(0.5 * np.array([[(self.alpha_list[i] ** 2) * (self.alpha_list[j] ** 2) * sqrt(
                    self.lambdasquare_list[i]) * sqrt(self.lambdasquare_list[j]) * (d - 1) * (((1 / (
                        (self.H_list[i] + self.H_list[j]) * (1 - (self.H_list[i] + self.H_list[j]) ** 2) * (
                        2 + self.H_list[i] + self.H_list[j]))) * (abs(tau + Delta) ** (
                            2 + self.H_list[i] + self.H_list[j]) - 2 * abs(tau) ** (2 + self.H_list[i] + self.H_list[
                    j]) + abs(tau - Delta) ** (2 + self.H_list[i] + self.H_list[j]))))
                                                   for i in
                                                   range(d)] for j in range(d)]))
            else:
                # sum_term = 1*np.sum(np.array([2 * (self.alpha_list[i] ** 2) * (self.alpha_list[j] ** 2) * sqrt(
                #     self.lambdasquare_list[i]) * sqrt(self.lambdasquare_list[j]) * (d - 2)* (2 * (
                #                                     gamma_2(i, j) / (self.H_list[i] + self.H_list[j] + 1) * tau ** (
                #                                     self.H_list[i] + self.H_list[j] + 1) / Delta)) for (i, j)
                #                             in zip(range(d), range(d))]))
                # sum_term = np.sum(np.array([[(self.alpha_list[i] ** 2) * (self.alpha_list[j] ** 2) * sqrt(
                #     self.lambdasquare_list[i]) * sqrt(self.lambdasquare_list[j]) * (d - 2) * (
                #                                      (self.H_list[i] + self.H_list[j] + 2) / (
                #                                      1 - self.H_list[i] - self.H_list[j])
                #                                      * tau ** (self.H_list[i] + self.H_list[j]) * Delta ** 2) for i in
                #                              range(d)] for j in range(d)]))
                # sum_term = np.sum(0.5*np.array([[(self.alpha_list[i] ** 2) * (self.alpha_list[j] ** 2) * sqrt(
                #     self.lambdasquare_list[i]) * sqrt(self.lambdasquare_list[j]) * (d - 2) * (1/ ( (self.H_list[i] + self.H_list[j])*(
                #                                      1 - self.H_list[i] - self.H_list[j]))
                #                                      * tau ** (self.H_list[i] + self.H_list[j]) * Delta ** 2) for i in
                #                              range(d)] for j in range(d)]))
                sum_term = np.sum(0.5 * np.array([[(self.alpha_list[i] ** 2) * (self.alpha_list[j] ** 2) * sqrt(
                    self.lambdasquare_list[i]) * sqrt(self.lambdasquare_list[j]) * (d - 2) * (((1 / (
                            (self.H_list[i] + self.H_list[j]) * (1 - (self.H_list[i] +self.H_list[j]) ** 2) * (
                                2 + self.H_list[i] + self.H_list[j])))*(abs(tau+Delta)**(2 + self.H_list[i] + self.H_list[j])-2*abs(tau)**(2 + self.H_list[i] + self.H_list[j])+abs(tau-Delta)**(2 + self.H_list[i] + self.H_list[j]))))
                                                   for i in
                                                   range(d)] for j in range(d)]))
            # return log(2 ** (q / 2) * special.gamma((q + 1) / 2) * (-sum_term)**(q / 2) / sqrt(pi))
            return log(2 ** (q / 2) * special.gamma((q + 1) / 2) * (sum_term) ** (q / 2) / sqrt(pi))
        q = 3
        Delta = 0.01
        tau_range_fromrange = np.linspace(10, 500, 100000)  # self.T * np.exp(logtauoverT_range)
        logtau_range = np.log(tau_range_fromrange)
        logtauoverT_range = np.log(tau_range_fromrange/ self.T ) #
        logtauoverDelta_range =  np.log(tau_range_fromrange/Delta)
        logMRM_Moments_values = [logMRM_Moments(q, tau, Delta) for tau in tau_range_fromrange]
        if method == 'linreglog moments':
            hurst_index_as_slope = np.polyfit(logtauoverT_range, logMRM_Moments_values, 1)[0]
            return hurst_index_as_slope / q
        if method == 'root finding':
            logtauoverT_range = np.linspace(0,5, 1000)
            tau_range_fromrange = self.T * np.exp(logtauoverT_range)
            def ObjectiveFunction(tau,H):
                z=Delta/tau
                factor = (2/3*log(sum(np.array(self.alpha_list)**2*np.exp(np.array(self.lambdasquare_list)*np.array(self.T_list)**(2*np.array(self.H_list))/(np.array(self.H_list)*(1-2*np.array(self.H_list)))))))
                g = (- 2 * abs(z) ** (2 * H ) ) / ( (2 * H + 1) * (2 * H + 2))
                #return ((logMRM_Moments(q, tau, Delta)) - log(2 ** (q / 2) / sqrt(pi) * special.gamma((q + 1) / 2))-(q / 2)*(log(-factor)+2 * H*log(2 * abs(z))-log((2 * H + 1) * (2 * H + 2))) -q*H*log(tau / self.T)) ** 2
                return ((logMRM_Moments(q, tau, Delta)) - log(2 ** (q / 2) / sqrt(pi) * special.gamma((q + 1) / 2)) - log(2 ** (q / 2) / sqrt(pi) * special.gamma((q + 1) / 2)*np.mean(self.lambdasquare_list)**q*tau**(q*H)/(H*(1-2*H))**(q/2)))
            objective_function = lambda H:np.mean([ObjectiveFunction(tau,H) for tau in tau_range_fromrange])
            # def ObjectiveFunctionPrime(tau,H):
            #     sum_term = np.sum(np.array([2 * (self.alpha_list[i] ** 2) * (self.alpha_list[j] ** 2) * sqrt(
            #         self.lambdasquare_list[i]) * sqrt(self.lambdasquare_list[j]) * (self.dimension - 1) * ((self.H_list[i] + self.H_list[j]+2)/(1-self.H_list[i] - self.H_list[j])
            #                                                                                   *tau**(self.H_list[i] + self.H_list[j])*Delta**2) for
            #                                 (i, j) in zip(range(self.dimension), range(self.dimension))]))
            #     g = (-3*sum_term/log(sum(np.array(self.alpha_list)**2*np.exp(np.array(self.lambdasquare_list)*np.array(self.T_list)**(2*np.array(self.H_list))/(np.array(self.H_list)*(1-2*np.array(self.H_list)))))))
            #     return (2*H*log(Delta/self.T)-log(2*H+1)-log(H+1)-log(g))**2
            # objective_function_prime = lambda H:np.mean([ObjectiveFunctionPrime(tau,H) for tau in tau_range_fromrange])
            # def ObjectiveFunctionAsympt(tau,H):
            #     z=Delta/tau
            #     factor = (2/3*log(sum(np.array(self.alpha_list)**2*np.exp(np.array(self.lambdasquare_list)*np.array(self.T_list)**(2*np.array(self.H_list))/(np.array(self.H_list)*(1-2*np.array(self.H_list)))))))
            #     g = -1
            #     return (exp(logMRM_Moments(q,tau,Delta))-(2 ** (q / 2) / sqrt(pi) * special.gamma((q + 1) / 2)*(factor*g) ** (q / 2) * (tau/self.T) ** (q *H)))**2
            # objective_function_Asympt = lambda H: np.mean( [ObjectiveFunctionAsympt(tau, H) for tau in tau_range_fromrange])

            plt.plot( np.linspace(1e-3,0.45,20), [objective_function(h) for h in  np.linspace(1e-3,0.45,20)] )
            plt.show()
            # print("objective_function_H_X^log=",optimize.newton(objective_function, x0=np.array(0.01),tol=1.48e-4,maxiter=100,disp=True))
            # newton_sol = optimize.newton(objective_function, x0=np.array(0.01),tol=1.48e-4,maxiter=100,disp=True)  # bounds=((1e-10), (0.45))
            brentq_sol = optimize.brentq(objective_function, 0.001,0.4,xtol=2e-12, rtol=8.881784197001252e-16, maxiter=100, full_output=True, disp=True)  # bounds=((1e-10), (0.45))
            return brentq_sol
        if method == 'linreglog moments with bias correction':
            # reg = LinearRegression().fit([[logtauoverT,logtauoverDelta] for logtauoverT,logtauoverDelta in zip(logtauoverT_range,logtauoverDelta_range)], logMRM_Moments_values)
            reg = LinearRegression().fit([[logtau,logtauoverDelta] for logtau,logtauoverDelta in zip(logtau_range,logtauoverDelta_range)], logMRM_Moments_values)
            print(" np.polyfit(logtauoverT_range, objective_function_values, 1)[0] = ", np.polyfit(logtauoverT_range, logMRM_Moments_values, 1)[0]/q)
            print("reg.coef_ = ",reg.coef_/q)
            # hurst_index_as_slope = reg.coef_[0]+1/2*reg.coef_[1]
            hurst_index_as_slope = reg.coef_[0] + reg.coef_[1]
            return hurst_index_as_slope / q
        if method == "GMM":
            def TheoreticalCovariance_Index(lag):
                H_list = -1 * np.log(1.0 / np.abs(np.array(self.H_list)) - 1)
                lambdasquare_list = -1 * np.log(1.0 / np.abs(np.array(self.lambdasquare_list)) - 1)
                # gamma_2 = lambda i,j: self.T**(self.H_list[i]+self.H_list[j])/((self.H_list[i]+self.H_list[j])*(1-(self.H_list[i]+self.H_list[j])**2))
                # gamma_1 = lambda i,j: 1/((self.H_list[i]+self.H_list[j])*(1-(self.H_list[i]+self.H_list[j])**2)*(2+self.H_list[i]+self.H_list[j]))
                # Gmm_Omega  = lambda i,j:gamma_2(i,j)-gamma_1(i,j)*lag**(self.H_list[i]+self.H_list[j])*(1+self.H_list[i]+self.H_list[j])*(2+self.H_list[i]+self.H_list[j])
                # if self.dimension%2==0:
                #     return 0.5*sum([self.alpha_list[i]**2*self.alpha_list[j]**2*sqrt(self.lambdasquare_list[i]*self.lambdasquare_list[j])*(self.dimension-1)/Delta**2*Gmm_Omega(i,j) for i,j in zip(range(self.dimension),range(self.dimension))])+np.sum(np.array(self.alpha_list)**4)
                # else:
                #     return 0.5*sum([self.alpha_list[i]**2*self.alpha_list[j]**2*sqrt(self.lambdasquare_list[i]*self.lambdasquare_list[j])*(self.dimension-2)/Delta**2*Gmm_Omega(i,j) for i,j in zip(range(self.dimension),range(self.dimension))])+np.sum(np.array(self.alpha_list)**4)
                gamma_2 = lambda i, j: self.T ** (H_list[i] + H_list[j]) / (
                            (H_list[i] +H_list[j]) * (1 - (H_list[i] + H_list[j]) ))
                gamma_1 = lambda i, j: 1 / (
                            (H_list[i] + H_list[j]) * (1 - (H_list[i] +H_list[j]) ** 2) * (
                                2 + H_list[i] + H_list[j]))

                cov_Omega = lambda i, j: (Delta**2*gamma_2(i, j) - gamma_1(i, j) * (abs(lag+Delta)**(2 +H_list[i] +H_list[j])-2*abs(lag)**(2 +H_list[i] +H_list[j])+abs(lag-Delta)**(2 +H_list[i] +H_list[j])))
                #cov_Omega = lambda i, j: Delta**2*(gamma_2(i, j) - gamma_1(i, j) * (lag ** (H_list[i] + H_list[j])) * ( 1 + H_list[i] + H_list[j]) * (2 +H_list[i] +H_list[j]))
                if self.dimension % 2 == 0:
                    return 0.5 * sum([self.alpha_list[i] ** 2 * self.alpha_list[j] ** 2 * sqrt(
                        lambdasquare_list[i] * lambdasquare_list[j]) * (
                                                  self.dimension - 1) *1/Delta**2 * cov_Omega(i, j) for i, j in
                                      zip(range(self.dimension), range(self.dimension))]) + np.sum(np.array(self.alpha_list) ** 4)
                else:
                    return 0.5 * sum([self.alpha_list[i] ** 2 * self.alpha_list[j] ** 2 * sqrt(
                        lambdasquare_list[i] * lambdasquare_list[j]) * (
                                                  self.dimension - 2)*1/Delta**2  * cov_Omega(i, j) for i, j in
                                      zip(range(self.dimension), range(self.dimension))]) + np.sum(np.array(self.alpha_list) ** 4)

            H_init = self.ComputeHurst_log_small_intermittencies('linreglog moments with bias correction')
            # if self.dimension % 2 == 0:
            #     variance_log_MRM_index = 0.5 * sum([self.alpha_list[i] ** 2 * self.alpha_list[j] ** 2 * sqrt(
            #         self.lambdasquare_list[i] * self.lambdasquare_list[j]) * (
            #                               self.dimension - 1) / Delta ** 2 * (self.T ** (self.H_list[i] + self.H_list[j]) / (
            #                 (self.H_list[i] +self.H_list[j]) * (1 - (self.H_list[i] + self.H_list[j]) ** 2))) for i, j in
            #                       zip(range(self.dimension), range(self.dimension))]) + np.sum(
            #         np.array(self.alpha_list) ** 4)
            # else:
            #     variance_log_MRM_index = 0.5 * sum([self.alpha_list[i] ** 2 * self.alpha_list[j] ** 2 * sqrt(
            #         self.lambdasquare_list[i] * self.lambdasquare_list[j]) * (
            #                               self.dimension - 2) / Delta ** 2 * (self.T ** (self.H_list[i] + self.H_list[j]) / (
            #                 (self.H_list[i] +self.H_list[j]) * (1 - (self.H_list[i] + self.H_list[j]) ** 2))) for i, j in
            #                       zip(range(self.dimension), range(self.dimension))]) + np.sum(
            #         np.array(self.alpha_list) ** 4)
            #
            # # lambdasquare_init = np.abs(variance_log_MRM_index * 2 * H_init * (1 - 2 * H_init))
            # lambdasquare_init = 0.01
            # print("H_init,lambdasquare_init = ",H_init,lambdasquare_init)
            GMM_index_Hlog = GMM()
            return GMM_index_Hlog.ComputeParamsGMM(TheoreticalCovariance_Index,H_Init=H_init, lambda2_Init=0.017) #0.01

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
                            2 / (3 * self.nu_inf_square) * log(A_d) + ( self.nu_inf_square) / (self.nu_sup_square) * self.T)}
            if check_boundedness == False:
                return {'Lowerbound': (1 / log(self.T ** 2)) * log(
                    2 / (3 *  self.nu_inf_square) * log(A_d) + (2 * self.nu_inf_square) / (3 * self.nu_sup_square)),
                        'Upperbound': (1 / log(self.T ** 2)) * log((self.nu_inf_square) / (self.nu_sup_square))+1/2}
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
        double_sum = 0
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i == j:
                    A_d += self.alpha_list[i] * self.alpha_list[j]
                elif i < j:
                    A_d += self.alpha_list[i] * self.alpha_list[j] * self.brownian_correlations[(i, j)]
                else:
                    A_d += self.alpha_list[i] * self.alpha_list[j] * self.brownian_correlations[(j, i)]
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i == j:
                    double_sum += self.nu_square * self.alpha_list[i] * self.alpha_list[j] / (
                            2 * A_d) * (self.nus_square_list[i] * self.T ** (2 * self.H_list[i]) +
                                          self.nus_square_list[i] * self.T ** (2 * self.H_list[j]))
                elif i < j:
                    double_sum += self.nu_square * self.alpha_list[i] * self.alpha_list[j] * \
                                  self.brownian_correlations[(i, j)] / (2 * A_d) * (
                                          self.nus_square_list[i] * self.T ** (2 * self.H_list[i]) +
                                          self.nus_square_list[i] * self.T ** (2 * self.H_list[j]))
                else:
                    double_sum += self.nu_square * self.alpha_list[i] * self.alpha_list[j] * \
                                  self.brownian_correlations[(j, i)] / (2 * A_d) * (
                                          self.nus_square_list[i] * self.T ** (2 * self.H_list[i]) +
                                          self.nus_square_list[i] * self.T ** (2 * self.H_list[j]))
        if brownian_correl_method == 'Brownian correlates - classical':
            if approximation_type == "T infty":
                return max(self.H_list)+log(self.nus_square_list[np.argmax(max(self.H_list))]/(self.nu_square))/log(self.T**2)
            if approximation_type == "Small intermittencies":
                return (1 / log(self.T ** 2)) * log(2 / (3 * self.nu_square) * (log(A_d) + double_sum))
        if brownian_correl_method == 'General case':
            if approximation_type == "Small intermittencies":
                # print("double sum check=",np.sum(self.nu_square * np.array(self.alpha_list)**2 / (A_d) * (np.array(self.nus_square_list) * self.T ** (2 * np.array(self.H_list)))))
                # print("log(A_d) + double_sum =", A_d,1/self.nu_square,double_sum,(2 / (3 * self.nu_square) * (log(A_d) + double_sum)))
                return (1 / log(self.T ** 2)) * log(2 / (3 * self.nu_square) * (log(A_d) + double_sum))

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






