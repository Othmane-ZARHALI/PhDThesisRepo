#  Paper:   "From Rough to Multifractal volatility: the log S-fBM model"
#  Purpose: Testing

# Author : Jean-Francois MUZY, Othmane ZARHALI


# Importations
import matplotlib.pyplot as plt
import numpy as np

from DataAcquisition import *
from LogSfbmModel import *
from GMMCalibration import *
from math import log

import warnings
warnings.filterwarnings("ignore")

#    DATA ACQUISITION AND PROCESSING ###################################################################################

# # Oxford Man institute realized vol acquisition and processing

File_path = "oxfordmanrealizedvolatilityindices.csv"
realized_vol_data_obj_ox = DataAcquisition('OxfordManInstitute',File_path)
# realized_vol_data_obj_ox.IndicesCharging()
# indices_list = realized_vol_data_obj_ox.indices_list
# test_index = indices_list[0]
# log_vol = realized_vol_data_obj_ox.GetlogVol(test_index,'bv',True,True,False)
# logvolvariance_overall_with_lag = realized_vol_data_obj_ox.GetlogVolVar_vs_Size(log_vol)
# means_and_variance_eachday_in_week = realized_vol_data_obj_ox.ComputeMeanVarianceinWeek(test_index)
# correl = Correlation(log_vol,log_vol)

logvol_synthesis_ox = realized_vol_data_obj_ox.LogVolSynthesisOverAssets(['.AEX','.AORD','.BFX', '.BSESN'])



# # Yahoo finance acquisition and processing

realized_vol_data_obj_yf = DataAcquisition('Yahoo finance')

realized_vol_data_obj_yf.IndicesCharging("AAPL",first_date="1900-01-01",last_date="2034-01-01")
# market_data = realized_vol_data_obj_yf.dataframe_indices
# market_capitalization = realized_vol_data_obj_yf.market_capitalization
# signal_test = market_data["Close"]
# signal_test = np.array(signal_test)
# removed0signal_test = realized_vol_data_obj_yf.removeZeros(signal_test)
log_vol_estimator = realized_vol_data_obj_yf.ComputeLogVolEstimator()
# CompleteData_withlogvol = realized_vol_data_obj_yf.GetCompleteData_withlogvol(symlist_SP)
# correlation_matrix_test = realized_vol_data_obj_yf.ComputeCovarianceMatrixLogvol(symlist_SP)[0]
# logvol_synthesis_yf = realized_vol_data_obj_yf.LogVolSynthesisOverAssets(["GOOGL","AAPL","AMZN"])


#    S FBM MODEL #######################################################################################################
# gaussian_process_test = GaussianProcess('LastWave',correlation_matrix_test[0])
# generated_gaussian_process = gaussian_process_test.Generate(10)
#
# S_fbm_model = Sfbm()
# Sfbmcorrelation = S_fbm_model.SfbmCorrelation(10,200)
# S_fbm_model_generation_example = S_fbm_model.GenerateSfbm(10)
# S_fbm_model_logvolgeneration_example = S_fbm_model.GeneratelogVol(200,10)

#   GMM CALIBRATION ####################################################################################################
GMM_obj = GMM()
# Simple checks
datamoments = GMM_obj.DataMoments(log_vol_estimator)

modelmoments = GMM_obj.ModelMoments(0.001, 2, 200, 0.2)
modelmoments_m =  GMM_obj.ModelMoments_m(0.001, 2, 200, 0.2)
modelmoments_M =  GMM_obj.ModelMoments_M(0.001, 2, 200, 0.2)

scaling_haar = GMM_obj.ScalingHaar(log_vol_estimator)

#hurstindexestimator = GMM_obj.HurstEstimator(log_vol_estimator)

# GMM estimates
#calibrated_parameters = GMM_obj.ComputeParamsGMM(log_vol_estimator)
#print(calibrated_parameters)

#calibrated_parameters_multipleassets_yf = GMM_obj.MultipleGMMCalibrations(logvol_synthesis_yf)
#calibrated_parameters_multipleassets_ox = GMM_obj.MultipleGMMCalibrations(logvol_synthesis_ox)

#print("calibrated_parameters_multipleassets = ",calibrated_parameters_multipleassets_ox)

# #    MULTIDIMENSIONAL S FBM MODEL ######################################################################################
# # Independent case
# MultidimS_fbm_model = MultidimensionalSfbm([Sfbm(H=0.03),Sfbm(H=0.03)])
# S_fbm_model_mutlidimensionalgeneration_example = MultidimS_fbm_model.GenerateMultidimensionalSfbm(4000)
# print("multi = ",S_fbm_model_mutlidimensionalgeneration_example)
# index_builder = MultidimS_fbm_model.Index_Builder([0.5,0.5],S_fbm_model_mutlidimensionalgeneration_example,'mrm and mrw')
# log_vol_index_generation_direct = MultidimS_fbm_model.GeneratelogVolMultidimSfbm_Index([0.5,0.5],'direct',4000)
#
# GMM_index = GMM()
# index_estimatedGMM_param = GMM_index.ComputeParamsGMM(log_vol_index_generation_direct)
# print("index_estimatedGMM_param = ",index_estimatedGMM_param)


# #    SYNTHETIC INDEX  ##################################################################################################
# # index = 0.25google+0.25aapl+0.5*amzn
# weights = [0.5,0.5]
# logvol_synthetic_index_yf = realized_vol_data_obj_yf.LogVolSyntheticIndexFromData(["AAPL","AMZN"],weights)
#
# GMM_obj_synthetic_index = GMM()
# synthtetic_index_paramGMM = GMM_obj_synthetic_index.ComputeParamsGMM(logvol_synthetic_index_yf)
# #print("synthtetic_index_paramGMM = ",synthtetic_index_paramGMM)

#    MULTIDIMENSIONAL S FBM MODEL ######################################################################################
# Independent case 2 dimensional
# MultidimS_fbm_model = MultidimensionalSfbm([Sfbm(H=0.03), Sfbm(H=0.03)])
# S_fbm_model_mutlidimensionalgeneration_example = MultidimS_fbm_model.GenerateMultidimensionalSfbm(4000)
# # print(S_fbm_model_mutlidimensionalgeneration_example)
# index_builder = MultidimS_fbm_model.Index_Builder([0.5, 0.5], S_fbm_model_mutlidimensionalgeneration_example,
#                                                   'mrm and mrw')
# log_vol_index_generation_direct = MultidimS_fbm_model.GeneratelogVolMultidimSfbm_Index([0.5, 0.5], 'direct', 4000)
#
# GMM_index = GMM()
# index_estimatedGMM_param = GMM_index.ComputeParamsGMM(log_vol_index_generation_direct)
# print("index_estimatedGMM_param = ",index_estimatedGMM_param)

# Independent case d dimensional
# 1 dimensional
# size = 4000
# H=0.1
# S_fbm_model = Sfbm(H,0.02,2**14) #T=0.732075  lambda2 = 0.068970
# # Sfbmcorrelation = S_fbm_model.SfbmCorrelation(size)
# S_fbm_model_generation_example = S_fbm_model.GenerateSfbm()
#
#
# S_fbm_model_logvolgeneration_example = S_fbm_model.GeneratelogVol(size)
# S_fbm_model_logvolgeneration_example_qv = S_fbm_model_logvolgeneration_example[1]   # direct computation IV estimate
#
# plt.plot(S_fbm_model_logvolgeneration_example_qv)
# plt.title( f'Sfbm model 1D - H={H} - log vol ')
# plt.show()
#
# GMM_1d = GMM()
# index_estimatedGMM_param1d = GMM_1d.ComputeParamsGMM(S_fbm_model_logvolgeneration_example_qv,15)
# print("index_estimatedGMM_param1d = ", index_estimatedGMM_param1d)


 # H estimated ~ 0.1



# dimension = 3
# H = 0.15
# Hs = [H for i in range(dimension)]
# weights = np.random.randint(1, 10, dimension)
#
# weights = weights / np.sum(weights)
#
# Sfbms = [Sfbm(Hs[i],0.068970 ,2**14) for i in range(dimension)]  #lambda2 = 0.068970  T=2**14
# MultidimensionalSfbms = MultidimensionalSfbm(Sfbms)
# Sfbms_generation_example = MultidimensionalSfbms.GenerateMultidimensionalSfbm(4000)
#
# index_builder_Sfbms = MultidimensionalSfbms.Index_Builder(weights, Sfbms_generation_example,'mrm and mrw')
# # print("index_builder_Sfbms = ",index_builder_Sfbms)
# log_vol_index_generation_direct_Sfbms = MultidimensionalSfbms.GeneratelogVolMultidimSfbm_Index(weights,'quadratic variation estimate',4000,'mrw')
#
# plt.plot(log_vol_index_generation_direct_Sfbms)
# plt.title( f'Sfbm model 1D index - H={H} - log vol ')
# plt.show()
# #
# GMM_index = GMM()
# index_estimatedGMM_paramSfbms = GMM_index.ComputeParamsGMM(log_vol_index_generation_direct_Sfbms,10)
# print("index_estimatedGMM_paramSfbms = ", index_estimatedGMM_paramSfbms)







# H distribution
# Number_indices = 5
# dimension = 3
# H = 0.15
# Hs = [H for i in range(dimension)]
# Multiple_weights,Multiple_Sfbms = [],[]
# Multiple_indices = dict()
# lambdasquare_list,T_list,sigma_list = [0.02 for i in range(dimension)],[2**14 for i in range(dimension)],[1 for i in range(dimension)]
#
#
# for i in range(Number_indices):
#     weights = np.random.randint(1, 10, dimension)
#     weights = weights / np.sum(weights)
#     Multiple_weights.append(weights)
#     Multiple_Sfbms.append([Sfbm(Hs[i],0.02 ,2**14) for i in range(dimension)])
#
# Multiple_Hs=[Hs for i in range(Number_indices)]
# Multiple_lambdasquare_list=[lambdasquare_list for i in range(Number_indices)]
# Multiple_T_list=[T_list for i in range(Number_indices)]
# Multiple_sigma_list=[sigma_list for i in range(Number_indices)]
# Multiple_correlations = [{} for i in range(Number_indices)]
#
# MultipleIndicesConstructor(Multiple_weights,Multiple_Sfbms,Multiple_correlations,Multiple_Hs,Multiple_lambdasquare_list,Multiple_T_list,Multiple_sigma_list)
#
# #MultipleIndicesConstructor_obj = MultipleIndicesConstructor(Multiple_weights,Multiple_Sfbms)
# MultipleIndicesConstructor_obj =MultipleIndicesConstructor(Multiple_weights,Multiple_Sfbms,Multiple_correlations,Multiple_Hs,Multiple_lambdasquare_list,Multiple_T_list,Multiple_sigma_list)
# trajectories_indices = MultipleIndicesConstructor_obj.ConstructIndicestrajectories(4000)
#
# keys = ['Index trajectory' for i in range(Number_indices)]
# log_vol_indices_dic = MultipleIndicesConstructor_obj.ConstructLogVolIndicestrajectories(4000,8,'direct',keys)
# Index_trajectories_synthesis = log_vol_indices_dic[1]
#
# print("Index_trajectories_synthesis = ",Index_trajectories_synthesis)
# plt.plot(Index_trajectories_synthesis['Index trajectory 0'])
# plt.title( f'Robustness test H estimation of {H} - log vol plot')
# plt.show()
#
# GMM_index_trajectories_obj = GMM()
# GMM_index_trajectories_obj.HurstIndexEvolution_GMMCalibration(Index_trajectories_synthesis,'histogram',"",10)



################   NON INDEPENDANT MUTLIDIMENSIONAL SfBM ###############################################################
# CLASSICAL APPROACH                                 #########################


# dimension = 3
# H = 0.05
# #Hs = [H for i in range(dimension)]
# Hs = [0.001236,0.064988 ,0.023365 ]
# # weights = np.random.randint(1, 10, dimension)
# # weights = weights / np.sum(weights)
# weights = np.array([0.2,0.7,0.1])
# lambdasquare_list,T_list,sigma_list = [0.02 for i in range(dimension)],[2**14 for i in range(dimension)],[1 for i in range(dimension)]
#
# Sfbms = []#[Sfbm(Hs[i],0.068970 ,2**14) for i in range(dimension)] #
#
# #correlations = {(0,1):0,(0,2):0,(0,3):0,(0,4):0,(1,2):0,(1,3):0,(1,4):0,(2,3):0,(2,4):0,(3,4):0}
# #correlations = {(0,1):0,(0,2):0,(1,2):0}
# correlations = {(0,1):-0.06368572,(0,2):0.9,(1,2):-0.13849021}  # => H ~ 0.33
# #correlations = {(0,1):0.2,(0,2):0.85,(1,2):0.01}
#
# MultidimensionalSfbms_generalmodel = MultidimensionalSfbm(Sfbms, correlations,dimension,Hs,lambdasquare_list,T_list,sigma_list )
# log_vol_index_generation_generalmodel_Sfbms = MultidimensionalSfbms_generalmodel.GeneratelogVolMultidimSfbm_Index(weights,'quadratic variation estimate',4000)
# print("log_vol_index_generation_generalmodel_Sfbms = ",log_vol_index_generation_generalmodel_Sfbms)
#
# plt.plot(log_vol_index_generation_generalmodel_Sfbms)
# plt.title( f'Sfbm model 1D index - H={H} - log vol ')
# plt.show()
#
# GMM_index = GMM()
# index_estimatedGMM_paramSfbms = GMM_index.ComputeParamsGMM(log_vol_index_generation_generalmodel_Sfbms,10)
# print("index_estimatedGMM_paramSfbms = ", index_estimatedGMM_paramSfbms)



# Robustness of the H index estimate


# Number_indices = 50
# dimension = 3
# H = 0.15
# #Hs = [H for i in range(dimension)]
# Hs = [0.001236,0.064988 ,0.023365 ]
# lambdasquare_list,T_list,sigma_list = [0.02 for i in range(dimension)],[2**14 for i in range(dimension)],[1 for i in range(dimension)]
#
#
# Multiple_weights,Multiple_Sfbms = [],[]
# Multiple_indices = dict()
# weights = np.array([0.2,0.7,0.1])
#
# for i in range(Number_indices):
#     # weights = np.random.randint(1, 10, dimension)
#     # weights = weights / np.sum(weights)
#     Multiple_weights.append(weights)
#     #Multiple_Sfbms.append([Sfbm(Hs[i],0.02 ,2**14) for i in range(dimension)])
#     Multiple_Sfbms.append([])
#
# Multiple_Hs=[Hs for i in range(Number_indices)]
# Multiple_lambdasquare_list=[lambdasquare_list for i in range(Number_indices)]
# Multiple_T_list=[T_list for i in range(Number_indices)]
# Multiple_sigma_list=[sigma_list for i in range(Number_indices)]
# correlations = {(0,1):-0.06368572,(0,2):0.9,(1,2):-0.13849021}
# Multiple_correlations = [correlations for i in range(Number_indices)]
#
# MultipleIndicesConstructor_obj = MultipleIndicesConstructor(Multiple_weights,Multiple_Sfbms,Multiple_correlations,Multiple_Hs,Multiple_lambdasquare_list,Multiple_T_list,Multiple_sigma_list)
#
# trajectories_indices = MultipleIndicesConstructor_obj.ConstructIndicestrajectories(4000)
#
# keys = ['Index trajectory' for i in range(Number_indices)]
# log_vol_indices_dic = MultipleIndicesConstructor_obj.ConstructLogVolIndicestrajectories(4000,8,'quadratic variation estimate',keys)
#
# Index_trajectories_synthesis = log_vol_indices_dic[1]
#
# # print("Index_trajectories_synthesis = ",Index_trajectories_synthesis)
# # plt.plot(Index_trajectories_synthesis['Index trajectory 0'])
# # plt.title( f'Robustness test H estimation of {H} - log vol plot')
# # plt.show()
#
# GMM_index_trajectories_obj = GMM()
# GMM_index_trajectories_obj.HurstIndexEvolution_GMMCalibration(Index_trajectories_synthesis,'histogram',"",10)


# RANDOM EIGEN VALUES BROWNIAN CORREL APPROACH      #########################

# dimension = 3
# H = 0.05
# #Hs = [H for i in range(dimension)]
# Hs = [0.001236,0.064988 ,0.023365 ]
# # weights = np.random.randint(1, 10, dimension)
# # weights = weights / np.sum(weights)
# weights = np.array([0.2,0.7,0.1])
# lambdasquare_list,T_list,sigma_list = [0.02 for i in range(dimension)],[2**14 for i in range(dimension)],[1 for i in range(dimension)]
#
# Sfbms = []#[Sfbm(Hs[i],0.068970 ,2**14) for i in range(dimension)] #
#
# #correlations = {(0,1):0,(0,2):0,(0,3):0,(0,4):0,(1,2):0,(1,3):0,(1,4):0,(2,3):0,(2,4):0,(3,4):0}
# #correlations = {(0,1):0,(0,2):0,(1,2):0}
# correlations = {(0,1):-0.06368572,(0,2):0.9,(1,2):-0.13849021}  # => H ~ 0.33
# #correlations = {(0,1):0.2,(0,2):0.85,(1,2):0.01}
# #
# MultidimensionalSfbms_generalmodel = MultidimensionalSfbm(Sfbms, correlations,dimension,Hs,lambdasquare_list,T_list,sigma_list )
# log_vol_index_generation_generalmodel_Sfbms = MultidimensionalSfbms_generalmodel.GeneratelogVolMultidimSfbm_Index(weights,'quadratic variation estimate',4000,'mrw',8,32,'Brownian correlates - random correl matrix')
# print("log_vol_index_generation_generalmodel_Sfbms = ",log_vol_index_generation_generalmodel_Sfbms)
#
# plt.plot(log_vol_index_generation_generalmodel_Sfbms)
# plt.title( f'Sfbm model 1D index - H={H} - log vol ')
# plt.show()
#
# GMM_index = GMM()
# index_estimatedGMM_paramSfbms = GMM_index.ComputeParamsGMM(log_vol_index_generation_generalmodel_Sfbms,10)
# print("index_estimatedGMM_paramSfbms = ", index_estimatedGMM_paramSfbms)




# Robustness of the H index estimate

# Number_indices = 50
# dimension = 3
# #H = 0.15
# #Hs = [H for i in range(dimension)]
# #Hs = [0.001236,0.064988 ,0.023365 ]
# Hs = [0,0,0]  # H_index non null
# lambdasquare_list,T_list,sigma_list = [0.02 for i in range(dimension)],[2**14 for i in range(dimension)],[1 for i in range(dimension)]
#
#
# Multiple_weights,Multiple_Sfbms = [],[]
# Multiple_indices = dict()
# weights = np.array([0.2,0.7,0.1])
#
# for i in range(Number_indices):
#     # weights = np.random.randint(1, 10, dimension)
#     # weights = weights / np.sum(weights)
#     Multiple_weights.append(weights)
#     #Multiple_Sfbms.append([Sfbm(Hs[i],0.02 ,2**14) for i in range(dimension)])
#     Multiple_Sfbms.append([])
#
# Multiple_Hs=[Hs for i in range(Number_indices)]
# Multiple_lambdasquare_list=[lambdasquare_list for i in range(Number_indices)]
# Multiple_T_list=[T_list for i in range(Number_indices)]
# Multiple_sigma_list=[sigma_list for i in range(Number_indices)]
# correlations = {(0,1):-0.06368572,(0,2):0.9,(1,2):-0.13849021}
# Multiple_correlations = [correlations for i in range(Number_indices)]
#
# MultipleIndicesConstructor_obj = MultipleIndicesConstructor(Multiple_weights,Multiple_Sfbms,Multiple_correlations,Multiple_Hs,Multiple_lambdasquare_list,Multiple_T_list,Multiple_sigma_list)
#
# trajectories_indices = MultipleIndicesConstructor_obj.ConstructIndicestrajectories(4000,8,'mrw','Brownian correlates - random correl matrix')
#
# keys = ['Index trajectory' for i in range(Number_indices)]
# log_vol_indices_dic = MultipleIndicesConstructor_obj.ConstructLogVolIndicestrajectories(4000,8,'quadratic variation estimate',keys,32,'Brownian correlates - random correl matrix')
#
# Index_trajectories_synthesis = log_vol_indices_dic[1]
#
# # print("Index_trajectories_synthesis = ",Index_trajectories_synthesis)
# # plt.plot(Index_trajectories_synthesis['Index trajectory 0'])
# # plt.title( f'Robustness test H estimation of {H} - log vol plot')
# # plt.show()
#
# GMM_index_trajectories_obj = GMM()
# GMM_index_trajectories_obj.HurstIndexEvolution_GMMCalibration(Index_trajectories_synthesis,'histogram',"",10)




# Increase the number of stocks - see the impact


# Number_indices = 50
# dimension = 5
# H = 0
# #Hs = [H for i in range(dimension)]
# #Hs = [0.001236,0.064988 ,0.023365 ]
# Hs = [0,0,0,0,0]  # H_index non null
# lambdasquare_list,T_list,sigma_list = [0.02 for i in range(dimension)],[2**14 for i in range(dimension)],[1 for i in range(dimension)]
#
#
# Multiple_weights,Multiple_Sfbms = [],[]
# Multiple_indices = dict()
# weights = np.array([0.2,0.45,0.1,0.15,0.1])
#
# for i in range(Number_indices):
#     # weights = np.random.randint(1, 10, dimension)
#     # weights = weights / np.sum(weights)
#     Multiple_weights.append(weights)
#     #Multiple_Sfbms.append([Sfbm(Hs[i],0.02 ,2**14) for i in range(dimension)])
#     Multiple_Sfbms.append([])
#
# Multiple_Hs=[Hs for i in range(Number_indices)]
# Multiple_lambdasquare_list=[lambdasquare_list for i in range(Number_indices)]
# Multiple_T_list=[T_list for i in range(Number_indices)]
# Multiple_sigma_list=[sigma_list for i in range(Number_indices)]
# correlations = {(0,1):-0.06,(0,2):0.9,(0,3):0.4,(0,4):0.02,(1,2):-0.7,(1,3):0.41,(1,4):-0.91,(2,3):-0.1,(2,4):-0.1,(3,4):0.95}
# Multiple_correlations = [correlations for i in range(Number_indices)]
#
# MultipleIndicesConstructor_obj = MultipleIndicesConstructor(Multiple_weights,Multiple_Sfbms,Multiple_correlations,Multiple_Hs,Multiple_lambdasquare_list,Multiple_T_list,Multiple_sigma_list)
#
# trajectories_indices = MultipleIndicesConstructor_obj.ConstructIndicestrajectories(4000,8,'mrw','Brownian correlates - random correl matrix')
#
# keys = ['Index trajectory' for i in range(Number_indices)]
# log_vol_indices_dic = MultipleIndicesConstructor_obj.ConstructLogVolIndicestrajectories(4000,8,'quadratic variation estimate',keys,32,'Brownian correlates - random correl matrix')
#
# Index_trajectories_synthesis = log_vol_indices_dic[1]
#
# print("Index_trajectories_synthesis = ",Index_trajectories_synthesis)
# plt.plot(Index_trajectories_synthesis['Index trajectory 0'])
# plt.title( f'Robustness test H estimation of {H} - log vol plot')
# plt.show()
#
# GMM_index_trajectories_obj = GMM()
# GMM_index_trajectories_obj.HurstIndexEvolution_GMMCalibration(Index_trajectories_synthesis,'histogram',"",10)





########################################################################################################################

# Theoretical formulas

from TheoreticalFormulas import *
dimension = 3
correlations = {(0,1):-0.06,(0,2):0.9,(0,3):0.4,(0,4):0.02,(1,2):-0.7,(1,3):0.41,(1,4):-0.91,(2,3):-0.1,(2,4):-0.1,(3,4):0.95}
correlations3={(0,1):0.06,(0,2):0.9,(1,2):0.7}
lambdasquare_list,T_list,sigma_list = [0.1 for i in range(dimension)],[2**100 for i in range(dimension)],[1 for i in range(dimension)] #
h_threashold =  1 / log((2**(11.5)) **2)
H_list = [0.01,0.02,0.03]
#H_list = [h_threashold,h_threashold,h_threashold]
alpha_list = [0.5,0.3,0.2]
hurst_index = VarIndexHurst(correlations3,H_list,alpha_list,lambdasquare_list,T_list,sigma_list)
#print("hurst_index = ",hurst_index.ComputeHurst())
# print("bounds = ",hurst_index.ComputeBounds())
#print("linear lower bounds = ",hurst_index.ComputeLinearLowerbound())  # available for high H_i's
# print("asympt T infty Hurst = ",hurst_index.ComputeFirstOrderApproximations())
#print("asympt small interm Hurst = ",hurst_index.ComputeFirstOrderApproximations('Brownian correlates - classical','Small intermittencies'))

g_i_j_matrix={(0,1):0.06,(0,2):0.9,(1,2):0.7}
#print("hurst_index _ general case model = ",hurst_index.ComputeHurst('General case',g_i_j_matrix))


# the intermittencies play an important role in hurst valuation

#plot wrt T
Number_indices = 100
Multiple_Hs=[H_list for i in range(Number_indices)]
Multiple_lambdasquare_list=[lambdasquare_list for i in range(Number_indices)]
T_values = np.linspace(50,200,Number_indices) #2**50
Multiple_T_list=[[T_value for i in range(3)] for T_value in T_values]
Multiple_sigma_list=[sigma_list for i in range(Number_indices)]
Multiple_alphas = [alpha_list for i in range(Number_indices)]
Multiple_correlations = [correlations3 for i in range(Number_indices)]

arguments = {'T_lists':Multiple_T_list,'correl_lists':Multiple_correlations,'alpha_lists':Multiple_alphas,'H_lists':Multiple_Hs,'lambda_square_lists':Multiple_lambdasquare_list,'sigma_lists':Multiple_sigma_list}
#hurst_index.ComputeEvolution('T without bounds',True,arguments)
# hurst_index.ComputeEvolution('T with bounds',True,arguments)

#plot wrt lambda2
# Number_indices_lambda = 50
# lambda_values = np.linspace(0.005,0.02,Number_indices_lambda)
# Multiple_lambda_values=[[lambda_value for i in range(3)] for lambda_value in lambda_values]
# Multiple_T_fixed_list=[[2**100 for j in range(3)] for i in range(Number_indices_lambda)]
# arguments_lambda = {'T_lists':Multiple_T_fixed_list,'correl_lists':Multiple_correlations,'alpha_lists':Multiple_alphas,'H_lists':Multiple_Hs,'lambda_square_lists':Multiple_lambda_values,'sigma_lists':Multiple_sigma_list}
# hurst_index.ComputeEvolution('Intermittencies',True,arguments_lambda)

#g_ij calibration
correlationscalib={(0,1):0.9,(0,2):0.9,(1,2):0.9}
lambdasquare_list,T_list,sigma_list = [0.07 for i in range(dimension)],[2**10 for i in range(dimension)],[1 for i in range(dimension)] #
hurst_index_calibration = VarIndexHurst(correlationscalib, [0.01,0.03,0.01],alpha_list,lambdasquare_list,T_list,sigma_list)  # H_i infuence g_ij
#print(Sfbm().GenerateShiftedMRM_sample(1,5,5))
#print(hurst_index.ComputeCrossMRMCovCurvature(5,0,1,10))
#print(hurst_index.ComputeSmoothCrossCov(0.05,0,2,10,"MRMCovCurvature_normalized"))
from time import time
#t=time()

#print(hurst_index.g_i_j_Calibration(0.005,0,2,100))
#print(hurst_index.g_i_j_Calibration(0.005,0,1,100))
#print(hurst_index.g_i_j_Calibration(0.005,1,2,100))
# g 02=-9.3384  nmc =100
# g 01=477915246233.9002  nmc =100
# g 12=-0.01073  nmc =100

#print(time()-t)

# // 36sec vs sequential 55.44 sec  nmc=10
#t=time()
#print(hurst_index.ComputeCrossMRMCovCurvature(0.005,0,1,100))
# print("time = ",time()-t) # 345 MC

#print(hurst_index_calibration.g_i_j_Calibration(0.005,0,1,100))


#print(hurst_index.g_i_j_Calibration(0.005,0,1,1000))
# print(hurst_index.g_i_j_Calibration(0.005,1,2,1000))



# print("log hurst = ",VarIndexHurst({(0,1):0,(0,2):0,(1,2):0},[0.01,0.041,0.05],[0.4,0.4,0.2],0.0001*np.ones(3),[2**10 for i in range(dimension)],sigma_list).ComputeHurst_log_small_intermittencies("root finding"))
 # intermittencies play important role as well as H_is (but the major part is from intermittencies)

Hvalues = np.linspace(1e-3,0.16,20) #.15
plt.plot(Hvalues,[VarIndexHurst({(0,1):0,(0,2):0,(1,2):0},h*np.ones(3),0.3*np.ones(3),0.0001*np.ones(3),[2**10 for i in range(dimension)],sigma_list).ComputeHurst_log_small_intermittencies("root finding") for h in Hvalues])
plt.xlabel('H')
plt.ylabel('$H^{log}_{X}$')
plt.title(r'$H^{log}_{X}$ evolution - constant hurst stocks')
plt.show()








