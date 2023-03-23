#  Paper:   "From Rough to Multifractal volatility: the log S-fBM model"
#  Purpose: Data acquisition of the market data

# Author : Jean-Francois MUZY, Othmane ZARHALI

# Importations
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import pandas as pd
#from numba import njit
import seaborn as sns


def FirstOrLast(the_iterable, condition = lambda x: True,FirstOrLastFlag = 'First'):
    if FirstOrLastFlag=="First":
        for i in the_iterable:
            if condition(i):
                return i
    if FirstOrLastFlag=="Last":
        i=0
        while condition(the_iterable[i]):
            i+=1
        return the_iterable[i]


def CovarianceOmega(t,s,H_i,H_j,T,lambda_i,lambda_j,g_i_j):
    return ((g_i_j*lambda_i*lambda_j)/((H_i+H_j)*(1-(H_i+H_j))))*(T**(H_i+H_j)-abs(t-s)**(H_i+H_j))

def BigCovMatrixOmega(t_min,t_max,H_list,T,lambda_i,lambda_j,g_i_j,N_maxtimestep):
    timestep = np.linspace(t_min,t_max,N_maxtimestep)
    BigCovMatrixOmega = [[0 for j in range(2*N_maxtimestep)] for i in range(2*N_maxtimestep)]
    for i in range(2 * N_maxtimestep):
        for j in range(i,2 * N_maxtimestep):
            if i==j:
                BigCovMatrixOmega[i][j] = CovarianceOmega(timestep[i // 2], timestep[j // 2], H_list[i % 2], H_list[j % 2], T,lambda_i, lambda_j,1)
            else:
                BigCovMatrixOmega[i][j]=CovarianceOmega(timestep[i//2],timestep[j//2],H_list[i%2],H_list[j%2],T,lambda_i,lambda_j,g_i_j)
    return np.array(BigCovMatrixOmega)+np.transpose(np.array(BigCovMatrixOmega))

def TwoDimentionalPathsSimulation(t_min,t_max,H_list,T,lambda_i,lambda_j,g_i_j_list,N_maxtimestep):
    big_covariance_matrix = BigCovMatrixOmega(t_min,t_max,H_list,T,lambda_i,lambda_j,g_i_j_list,N_maxtimestep)
    choleskymatrix = np.linalg.cholesky(big_covariance_matrix)
    complete_path = choleskymatrix@np.random.normal(loc=0.0, scale=1.0, size=2*N_maxtimestep)
    return np.array([complete_path[i] for i in range(0,2*N_maxtimestep,2)]),np.array([complete_path[i] for i in range(1,2*N_maxtimestep,2)])

def g_i_j_calibration(tau,Delta,H_list,T,lambda_i,lambda_j,g_i_j_list,N_maxtimestep,N_maxMC,Flag_variance_reduction = False):
    gamma_1_ij = 1/((H_list[0]+H_list[1])*(1-(H_list[0]+H_list[1])**2)*(2+(H_list[0]+H_list[1])))
    gamma_2_ij = T**(H_list[0] + H_list[1]) / ((H_list[0] + H_list[1]) * (1 - (H_list[0] + H_list[1])))
    MC_copies_i,MC_copies_j = [],[]
    timescale = np.linspace(0, tau + Delta, N_maxtimestep)
    log_MRM_j_min_index = list(timescale).index(FirstOrLast(timescale,lambda x: x > Delta,'First'))
    log_MRM_i_max_index = log_MRM_j_min_index - 1
    if Flag_variance_reduction == False:
        for k in range(N_maxMC):
            log_MRM_i = TwoDimentionalPathsSimulation(0, Delta, H_list, T, lambda_i, lambda_j, g_i_j_list, N_maxtimestep)[0]
            log_MRM_j = TwoDimentionalPathsSimulation(Delta,tau+Delta, H_list, T, lambda_i, lambda_j, g_i_j_list, N_maxtimestep)[1]
            # log_MRM_i,log_MRM_j = log_MRM_i[:log_MRM_i_max_index]-np.mean(log_MRM_i[:log_MRM_i_max_index]),log_MRM_j[log_MRM_j_min_index:]-np.mean(log_MRM_j[log_MRM_j_min_index:])
            # log_MRM_i, log_MRM_j = log_MRM_i - np.mean(log_MRM_i), log_MRM_j- np.mean(log_MRM_j)

            # print("np.mean(log_MRM_i),np.mean(log_MRM_j) =", np.mean(log_MRM_i),np.mean(log_MRM_j))
            # print("log_MRM_i =",log_MRM_i)
            # print("log_MRM_j =", log_MRM_j)

            # print("log_MRM_i.sum(),log_MRM_j.sum() = ",log_MRM_i.sum(),log_MRM_j.sum())
            # MC_copies_i.append(np.cumsum(log_MRM_i, axis=0)[-1] * (Delta) / log_MRM_j_min_index)
            # MC_copies_j.append(np.cumsum(log_MRM_j, axis=0)[-1] * (tau + Delta) / N_maxtimestep)
            MC_copies_i.append(log_MRM_i.sum() * (Delta) / N_maxtimestep)
            MC_copies_j.append(log_MRM_j.sum() * (Delta) / N_maxtimestep)
            # print("1/lambda_i*log_MRM_i.sum() * (Delta) / N_maxtimestep = ",1/lambda_i*log_MRM_i.sum() * (Delta) / N_maxtimestep,log_MRM_i.sum() * (Delta) / N_maxtimestep)
        MC_copies_i, MC_copies_j =np.array(MC_copies_i),np.array(MC_copies_j)
        # print("np.sum((1/(lambda_i*lambda_j))*MC_copies_i*MC_copies_j) = ",1/N_maxMC*np.sum(MC_copies_i*MC_copies_j),1/N_maxMC*np.sum((1/(lambda_i*lambda_j))*MC_copies_i*MC_copies_j))
        # print("1/(lambda_i*lambda_j) = ",1/(lambda_i*lambda_j))
        # print("result1 = ",1/N_maxMC*np.sum((1/(lambda_i*lambda_j))*MC_copies_i*MC_copies_j)/(gamma_2_ij*Delta**2-gamma_1_ij*Delta**2*tau**(H_list[0] + H_list[1])*(2+(H_list[0]+H_list[1]))*(1+(H_list[0]+H_list[1]))))
        # print("result2=",1/N_maxMC*np.sum(MC_copies_i*MC_copies_j)/(gamma_2_ij*Delta**2-gamma_1_ij*Delta**2*tau**(H_list[0] + H_list[1])*(2+(H_list[0]+H_list[1]))*(1+(H_list[0]+H_list[1]))))
        estimate = 1/N_maxMC*np.sum((1/(lambda_i*lambda_j))*MC_copies_i*MC_copies_j)/(gamma_2_ij*Delta**2-gamma_1_ij*Delta**2*tau**(H_list[0] + H_list[1])*(2+(H_list[0]+H_list[1]))*(1+(H_list[0]+H_list[1])))
        variance = np.var((1/(lambda_i*lambda_j))*MC_copies_i*MC_copies_j/(gamma_2_ij*Delta**2-gamma_1_ij*Delta**2*tau**(H_list[0] + H_list[1])*(2+(H_list[0]+H_list[1]))*(1+(H_list[0]+H_list[1]))))
        return estimate,sqrt(variance/N_maxMC)
    if Flag_variance_reduction == True:
        MC_copies_i_minus,MC_copies_j_minus = [],[]
        for k in range(N_maxMC):
            log_MRM_i,log_MRM_i_minus = TwoDimentionalPathsSimulation(0, Delta, H_list, T, lambda_i, lambda_j, g_i_j_list, N_maxtimestep)[0],TwoDimentionalPathsSimulation(0, Delta, H_list, T, lambda_i, lambda_j, g_i_j_list, N_maxtimestep)[0]
            log_MRM_j,log_MRM_j_minus = TwoDimentionalPathsSimulation(Delta, tau + Delta, H_list, T, lambda_i, lambda_j, g_i_j_list, N_maxtimestep)[1],TwoDimentionalPathsSimulation(Delta, tau + Delta, H_list, T, lambda_i, lambda_j, g_i_j_list, N_maxtimestep)[1]
            # print("log_MRM_i_minus = ",log_MRM_i_minus)
            # print("log_MRM_i = ",log_MRM_i)
            MC_copies_i.append(log_MRM_i.sum() * (Delta) / N_maxtimestep)
            MC_copies_j.append(log_MRM_j.sum() * (Delta) / N_maxtimestep)
            MC_copies_i_minus.append(log_MRM_i_minus.sum() * (Delta) / N_maxtimestep)
            MC_copies_j_minus.append(log_MRM_j_minus.sum() * (Delta) / N_maxtimestep)
        MC_copies_i, MC_copies_j, MC_copies_i_minus, MC_copies_j_minus = np.array(MC_copies_i), np.array(MC_copies_j),np.array(MC_copies_i_minus), np.array(MC_copies_j_minus)
        estimate = 1 / (N_maxMC) * ((1 / (lambda_i * lambda_j))*(np.sum(MC_copies_i * MC_copies_j)+np.sum(MC_copies_i_minus * MC_copies_j_minus))/2) / (
                    gamma_2_ij * Delta ** 2 - gamma_1_ij * Delta ** 2 * tau ** (H_list[0] + H_list[1]) * (
                        2 + (H_list[0] + H_list[1])) * (1 + (H_list[0] + H_list[1])))
        variance = np.var((1 / (lambda_i * lambda_j)) * ((MC_copies_i * MC_copies_j+MC_copies_i_minus * MC_copies_j_minus)/2) / (
                    gamma_2_ij * Delta ** 2 - gamma_1_ij * Delta ** 2 * tau ** (H_list[0] + H_list[1]) * (
                        2 + (H_list[0] + H_list[1])) * (1 + (H_list[0] + H_list[1]))))
        return estimate, sqrt(variance / N_maxMC)


#print(CovarianceOmega(1,2,0.01,0.02,5,0.01,0.01,0.01))
#print(BigCovMatrixOmega(1,2,[0.01,0.02],5,0.01,0.01,0.01,50))
#print(TwoDimentionalPathsSimulation(1,2,[0.01,0.02],5,0.01,0.01,0.01,50))
#print(g_i_j_calibration(100,5,[0.01,0.02],150,0.001,0.001,0.01,200,1000))
print(g_i_j_calibration(100,5,[0.01,0.02],150,0.001,0.001,0.01,4000,100000,True))


def g_i_j_calibrationStatistics(tau,Delta,H_list,T,lambda_i,lambda_j,g_i_j_list,N_maxtimestep,N_maxMC,N_MCestim,outputtype='Mean'):
    S = []
    for k in range(N_MCestim):
        S.append(g_i_j_calibration(tau, Delta, H_list, T, lambda_i, lambda_j, g_i_j_list, N_maxtimestep, N_maxMC))
    if outputtype=='Mean':
        return 1 / N_MCestim * sum(S)
    if outputtype=='Histogram':
        plt.hist(S, bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram of g_ij estimate")
        plt.show()
        return "plotted"



print(g_i_j_calibrationStatistics(100,5,[0.01,0.02],150,0.001,0.001,0.01,4000,100000,10,"Histogram"))


