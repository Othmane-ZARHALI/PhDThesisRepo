#  Paper:   "From Rough to Multifractal volatility: the log S-fBM model"
#  Purpose: Testing

# Author : Jean-Francois MUZY, Othmane ZARHALI


# Importations
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

realized_vol_data_obj_yf.IndicesCharging("GOOGL",first_date="1900-01-01",last_date="2034-01-01")
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
# #print(S_fbm_model_mutlidimensionalgeneration_example)
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
# size = 4000
# S_fbm_model = Sfbm(0.01,0.068970 ,log(2.079392))
# Sfbmcorrelation = S_fbm_model.SfbmCorrelation(size)
# S_fbm_model_generation_example = S_fbm_model.GenerateSfbm(size)
# S_fbm_model_logvolgeneration_example = S_fbm_model.GeneratelogVol(size)
# S_fbm_model_logvolgeneration_example_qv = S_fbm_model_logvolgeneration_example[1]
# GMM_1d = GMM()
# index_estimatedGMM_param1d = GMM_1d.ComputeParamsGMM(S_fbm_model_logvolgeneration_example_qv,10,LagSignal=np.array([1, 2, 4, 5, 8, 11, 16, 22, 32, 45, 64, 90, 128]),
#                          GMM_Method=1)
# print("index_estimatedGMM_param1d = ", index_estimatedGMM_param1d)




dimension = 2
Hs = [0.25 for i in range(dimension)]
weights = np.random.randint(1, 10, dimension)

weights = weights / np.sum(weights)
#
Sfbms = [Sfbm(H=Hs[i]) for i in range(dimension)]
# MultidimensionalSfbms = MultidimensionalSfbm(Sfbms)
# Sfbms_generation_example = MultidimensionalSfbms.GenerateMultidimensionalSfbm(4000)
# index_builder_Sfbms = MultidimensionalSfbms.Index_Builder(weights, Sfbms_generation_example,'mrm and mrw')
# print("index_builder_Sfbms = ",index_builder_Sfbms)
# log_vol_index_generation_direct_Sfbms = MultidimensionalSfbms.GeneratelogVolMultidimSfbm_Index(weights,'quadratic variation estimate',4000)
#
# GMM_index = GMM()
# index_estimatedGMM_paramSfbms = GMM_index.ComputeParamsGMM(log_vol_index_generation_direct_Sfbms,10)
# print("index_estimatedGMM_paramSfbms = ", index_estimatedGMM_paramSfbms)

# H distribution
Number_indices = 15
dimension = 2
Hs = [0.25 for i in range(dimension)]
Multiple_weights,Multiple_Sfbms = [],[]
Multiple_indices = dict()
for i in range(Number_indices):
    weights = np.random.randint(1, 10, dimension)
    weights = weights / np.sum(weights)
    Multiple_weights.append(weights)
    Multiple_Sfbms.append([Sfbm(H=Hs[i]) for i in range(dimension)])


MultipleIndicesConstructor_obj = MultipleIndicesConstructor(Multiple_weights,Multiple_Sfbms)
trajectories_indices = MultipleIndicesConstructor_obj.ConstructIndicestrajectories(4000)
#print("trajectories_indices = ",trajectories_indices)

log_vol_indices_dic = MultipleIndicesConstructor_obj.ConstructLogVolIndicestrajectories(4000,4,'quadratic variation estimate',['Index trajectory' for i in range(Number_indices)])
Index_trajectories_synthesis = log_vol_indices_dic[1]

GMM_index_trajectories_obj = GMM()
print(GMM_index_trajectories_obj.MultipleGMMCalibrations(Index_trajectories_synthesis))

GMM_index_trajectories_obj.HurstIndexEvolution_GMMCalibration(Index_trajectories_synthesis,'curve')


