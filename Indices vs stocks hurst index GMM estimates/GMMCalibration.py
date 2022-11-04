#  Paper:   "From Rough to Multifractal volatility: the log S-fBM model"
#  Purpose: GMM calibration of LogSfbm

# Author : Jean-Francois MUZY, Othmane ZARHALI


# Importations
import numpy as np
from statsmodels.sandbox.regression.gmm import GMM
import matplotlib.pylab as plt
import scipy.optimize as opt
import scipy.special as sc
from scipy.stats import chi2

import numpy.linalg as lin

from DataAcquisition import *

#### Computation of the correlation function using FFT
from scipy.signal import correlate




class GMM:
    def __init__(self):
        pass

    #RETURNS: array of moments corresponding to each chosen lag
    def DataMoments(self,data_stream, LagSignal=[0, 1, 2, 3, 4, 5, 6]):
        lagMax = max(LagSignal)
        mean_data = data_stream.mean()
        _, cc = Correlation(data_stream - mean_data, data_stream - mean_data, dMin=0, dMax=lagMax)
        LagSignalInt = [int(lag) for lag in LagSignal]
        moments = cc[LagSignalInt]
        return moments

    def ModelMoments(self,H, lambda2, T, lsigma2, Delta=1, LagSignal=[0, 1, 2, 3, 4, 5, 6], flagLambda2=True):
        '''
        --------------------------------------------------------------------
        This function returns model correlation values as a function of 
        the model parameters
        --------------------------------------------------------------------
        INPUTS:
        lambda2  = intermittency coeff
        lsigma2  = noise variance
        T = Lage scale (in log)
        H = Hurst exponent: If fact the Hurst exponent is HH = 1/(1+exp(-H))
        LagSignal  = list (or array) of lags
        flagHighFreq = Flase => Uses Low-freq formula
                     = True  => Uses High-freq formula

        flagLambda2  = True  => Estimates lambda2
                     = False => Estimates nu2 = lambda2/(H)(1-2*H)

        RETURNS: Theoretical covariance values
        ------------------------------------------------------------------
        '''
        LagSignal = np.array(LagSignal)
        dSig = (LagSignal == 0).astype('float')
        HH = 1 / (1 + np.exp(-1 * H))
        if (flagLambda2):
            LL = 1 / (1 + np.exp(-1 * lambda2))
        else:
            LL = np.exp(lambda2)
        LS = np.exp(lsigma2)
        ee1 = 2 * HH + 1
        ee2 = 2 * HH + 2
        r1 = (np.abs(LagSignal + Delta) ** ee2 + np.abs(
            LagSignal - Delta) ** ee2 - 2 * LagSignal ** ee2) / ee2 / ee1 / Delta / Delta
        if (flagLambda2):
            KH2 = 1 / (2 * HH) / (1 - 2 * HH)
        else:
            KH2 = 1 / 2.0
        mm = LL * KH2 * (np.exp(T) - r1)
        mm[mm < 0] = 0
        mm = mm + LS * dSig
        return mm

    def ModelMoments_m(self,H, lambda2, lsigma2, Delta=1, LagSignal=[0, 1, 2, 3, 4, 5, 6], qval=2, flagLambda2=True):
        '''
        --------------------------------------------------------------------
        This function returns model correlation values as a function of
        the model parameters
        --------------------------------------------------------------------
        INPUTS:
        lambda2  = intermittency coeff
        lsigma2  = noise variance
        T = Lage scale (in log)
        H = Hurst exponent: If fact the Hurst exponent is HH = 1/(1+exp(-H))
        lagSig  = list (or array) of lags
        flagHighFreq = Flase => Uses Low-freq formula
                     = True  => Uses High-freq formula

        flagLambda2  = True  => Estimates lambda2
                     = False => Estimates nu2 = lambda2/(H)(1-2*H)

        RETURNS: Theoretical covariance values
        ------------------------------------------------------------------
        '''
        LagSignal = np.array(LagSignal)
        dSig = (LagSignal == 0).astype('float')
        HH = 1 / (1 + np.exp(-1 * H))
        #
        if (flagLambda2):
            LL = 1 / (1 + np.exp(-1 * lambda2))
        else:
            LL = np.exp(lambda2)

        # LS = lsigma2
        LS = np.exp(lsigma2)
        # ee0 = 2*HH
        ee1 = 2 * HH + 1
        ee2 = 2 * HH + 2
        # eem1 = 1-2*HH
        r1 = (np.abs(LagSignal + Delta) ** ee2 + np.abs(
            LagSignal - Delta) ** ee2 - 2 * LagSignal ** ee2) / ee2 / ee1 / Delta / Delta
        if (flagLambda2):
            KH2 = 1.0 / HH / (1 - 2 * HH)
        else:
            KH2 = 1.0
        mm = LL * KH2 * r1 + LS
        return mm

    def ModelMoments_M(self,H, lambda2, K1, T, Delta=1, LagSignal=[1, 2, 3, 4, 5, 6], qval=2, flagLambda2=True):
        '''
        --------------------------------------------------------------------
        This function returns model correlation values as a function of
        the model parameters
        --------------------------------------------------------------------
        INPUTS:
        lambda2  = intermittency coeff
        lsigma2  = noise variance
        T = Lage scale (in log)
        K = A constant to be adjusted (in log)
        H = Hurst exponent: If fact the Hurst exponent is HH = 1/(1+exp(-H))
        lagSig  = list (or array) of lags

        RETURNS: Theoretical covariance values
        ------------------------------------------------------------------
        '''
        FF = lambda Hurst,x : sc.hyp1f1(1.,1+1.0/(2*Hurst),x)-0.5*sc.hyp1f1(1.,1+1.0/Hurst,x)
        lagSig = np.array(LagSignal)
        lagSig1 = np.append(0, lagSig)
        HH = 1 / (1 + np.exp(-1 * H))
        if (flagLambda2):
            LL = 1 / (1 + np.exp(-1 * lambda2))
        else:
            LL = np.exp(lambda2)
        if (HH > 0.001):
            if (flagLambda2):
                K2 = LL / (2 * HH * (1 - 2 * HH))
            else:
                K2 = LL / 2.0
            TT = np.exp(2 * HH * T)
            KK1 = np.exp(K1)
            xx1 = np.abs(lagSig1 + Delta)
            xx2 = np.abs(lagSig1 - Delta)
            xx3 = np.abs(lagSig1)
            F1 = xx1 * xx1 * np.exp(K2 * (K1 - xx1 ** (2 * HH))) * FF(HH, K2 * (xx1 ** (2 * HH)))
            F2 = xx2 * xx2 * np.exp(K2 * (K1 - xx2 ** (2 * HH))) * FF(HH, K2 * (xx2 ** (2 * HH)))
            F3 = xx3 * xx3 * np.exp(K2 * (K1 - xx3 ** (2 * HH))) * FF(HH, K2 * (xx3 ** (2 * HH)))
        else:
            KK1 = np.exp(K1)
            TT = np.exp(LL * T)
            ee1 = 2 - LL
            A1 = 1 / (2 - LL) / (1 - LL)
            A1 *= TT
            F3 = A1 * lagSig1 ** (ee1)
            F1 = A1 * np.abs(lagSig1 + Delta) ** (ee1)
            F2 = A1 * np.abs(lagSig1 - Delta) ** (ee1)
        mm = KK1 * (F1 + F2 - 2 * F3)
        return (mm[1:])

    def ScalingHaar(self,signal, lagSignal=[1, 2, 4, 8, 16, 32]):
        scaling_haar = np.zeros(len(lagSignal))
        #zzz = np.cumsum(np.exp(signal))
        zzz = np.cumsum(signal)
        for i in range(len(lagSignal)):
            scale = int(lagSignal[i])
            xxx = (zzz[scale:] - zzz[:-scale])
            # xxx1 = np.log(xxx[scale:])-np.log(xxx[:-scale])
            xxx1 = (xxx[scale:] - xxx[:-scale]) / scale
            scaling_haar[i] = xxx1.std()
        return lagSignal, scaling_haar

    def HurstEstimator(self,datastream_sample, lagSignal=[1, 2, 4, 8, 16, 32]):
        if (lagSignal[0] < 1):
            lagSignal = lagSignal[1:]
        exp_datastream_sample = np.exp(datastream_sample)
        exp_datastream_sample_cumsum = np.cumsum(exp_datastream_sample)  #, axis=1
        zz = np.zeros(len(lagSignal))
        for i in range(len(lagSignal)):
            scale = int(lagSignal[i])
            #xxx = exp_datastream_sample_cumsum[:, scale:] - exp_datastream_sample_cumsum[:, :-scale]
            xxx = exp_datastream_sample_cumsum[scale:] - exp_datastream_sample_cumsum[:-scale]
            xxx = np.log(xxx)
            #zz[i] = (np.abs(xxx[:, scale:] - xxx[:, :-scale])).mean()
            zz[i] = (np.abs(xxx[scale:] - xxx[:-scale])).mean()
        H = max(np.polyfit(np.log(lagSignal), np.log(zz), deg=1)[0], 0.001)
        #l2 = (datastream_sample[:, 1:] - datastream_sample[:, :-1]).var() * 2 * H * (1 - 2 * H)
        if len(datastream_sample.shape)==1:
            lambda2 = np.abs((datastream_sample[1:] - datastream_sample[:-1]).var() * 2 * H * (1 - 2 * H))
        else:
            lambda2 = np.abs((datastream_sample[0][1:] - datastream_sample[0][:-1]).var() * 2 * H * (1 - 2 * H))
        # print("hurt index estimator, datastream_sample = ", datastream_sample[0])
        # print("hurt index estimator, H = ", H)
        # print("hurt index estimator, lambda2 = ",lambda2)
        # print('"""""""""""""""""""""""""""""""""""""""""')
        return H, lambda2

    def ErrorVecSignal(self,logvol_samples, H, lambda2, T, lsigma2=-10, lagSignal=[1, 2, 4, 8, 10], flagErr=True, flagLambda2=True):
        '''
        --------------------------------------------------------------------
        This function computes the vector of moment errors
        --------------------------------------------------------------------
        INPUTS:
        xvals = log-volatility (sample) of length L
        lambda2  = intermittency coeff
        lsigma2  = Variance of log of the noise (in case of MRW)
        T = Large scale (in log)
        H = Hurst exponent --> HH = 1/(1/exp(-H))
        lagSig  = list (or array) of lags

        flagErr = Boolean: True -> Errors
                            False -> Correlation
        RETURNS: err_vec of the covariance of xvals
        --------------------------------------------------------------------
        '''

        if (len(logvol_samples.shape) == 1):
            logvol_samples = np.array([logvol_samples])
        if (flagErr):
            #moments_model = self.ModelMoments(H=H, lambda2=lambda2, T=T, lsigma2=lsigma2, lagSig=lagSignal,flagLambda2=flagLambda2)
            moments_model = self.ModelMoments(H=H, lambda2=lambda2, T=T, lsigma2=lsigma2, Delta = 1, LagSignal=lagSignal,flagLambda2=flagLambda2)
        else:
            moments_model = np.zeros(len(lagSignal))
        lagMax = int(np.max(lagSignal))
        ### Let us padd xm with mean values
        xmz = np.zeros(logvol_samples.shape[1])
        xm = np.zeros((logvol_samples.shape[0], 2 * logvol_samples.shape[1]))
        for i in range(xm.shape[0]):
            # xm[i,:] = np.append(xvals[i,:]-xvals[i,:].mean(),xmz)
            xm[i, :] = np.append(logvol_samples[i, :] - logvol_samples.mean(), xmz)
        xm_size = logvol_samples.shape[1]

        ### We dont want to subtract theoretical error
        m1 = (xm[:, 0:xm_size] * xm[:, lagSignal[0]:lagSignal[0] + xm_size]).mean(axis=0) - moments_model[0]
        m2 = (xm[:, 0:xm_size] * xm[:, lagSignal[1]:lagSignal[1] + xm_size]).mean(axis=0) - moments_model[1]
        mm = np.vstack((m1, m2))
        for i in range(2, len(lagSignal)):
            m3 = (xm[:, 0:xm_size] * xm[:, lagSignal[i]:lagSignal[i] + xm_size]).mean(axis=0) - moments_model[i]
            mm = np.vstack((mm, m3))
        return np.array(mm)

    def ErrorVecSignal_m(self,logvol_samples, H, lambda2, lsigma2=-10, lagSignal=[1, 2, 4, 8, 10], qval=2, flagErr=True, flagLambda2=True):
        '''
        --------------------------------------------------------------------
        This function computes the vector of moment errors
        --------------------------------------------------------------------
        INPUTS:
        xvals = log-volatility (sample) of length L
        lambda2  = intermittency coeff
        lsigma2  = Variance of log of the noise (in case of MRW)
        T = Large scale (in log)
        H = Hurst exponent --> HH = 1/(1/exp(-H))
        lagSig  = list (or array) of increment scales
        qval = Moment order
        flagErr = Boolean: True -> Errors
                            False -> Correlation
        RETURNS: err_vec of the moments of order qval of M = exp(xvals)
        --------------------------------------------------------------------
        '''

        ### Compute Theoretical expressions:

        if (len(logvol_samples.shape) == 1):
            logvol_samples = np.array([logvol_samples])

        moms_model_m = self.ModelMoments_m(H, lambda2, lsigma2, Delta=1, LagSignal=lagSignal, qval=qval, flagLambda2=flagLambda2)

        ### We dont want to subtract theoretical error
        if not (flagErr):
            moms_model_m = np.zeros(len(lagSignal))

        lagMax = int(np.max(lagSignal))
        xm = logvol_samples
        xm_size = xm.shape[1]

        m1 = (np.abs(xm[:, lagSignal[0]:xm_size + lagSignal[0] - lagMax] - xm[:, :xm_size - lagMax]) ** qval).mean(axis=0) - \
             moms_model_m[0]
        m2 = (np.abs(xm[:, lagSignal[1]:xm_size + lagSignal[1] - lagMax] - xm[:, :xm_size - lagMax]) ** qval).mean(axis=0) - \
             moms_model_m[1]
        mm = np.vstack((m1, m2))
        for i in range(2, len(lagSignal)):
            m3 = (np.abs(xm[:, lagSignal[i]:xm_size + lagSignal[i] - lagMax] - xm[:, :xm_size - lagMax]) ** qval).mean(
                axis=0) - moms_model_m[i]
            mm = np.vstack((mm, m3))
        return np.array(mm)


    def ErrorVecSignal_M(self,logvol_samples, H, lambda2, K1, T, lagSignal=[1, 2, 4, 8, 10], flagErr=True, flagLambda2=True):
        '''
        --------------------------------------------------------------------
        This function computes the vector of moment errors
        --------------------------------------------------------------------
        INPUTS:
        xvals = log-volatility (sample) of length L
        lambda2  = intermittency coeff
        lsigma2  = Variance of log of the noise (in case of MRW)
        T = Large scale (in log)
        H = Hurst exponent --> HH = 1/(1/exp(-H))
        lagSig  = list (or array) of lags

        flagErr = Boolean: True -> Errors
                            False -> Correlation
        RETURNS: err_vec for correlation of M = exp(xvals)
        --------------------------------------------------------------------
        '''

        if (len(logvol_samples.shape) == 1):
            logvol_samples = np.array([logvol_samples])

        ### We thake the exponential of xvals
        xvExp = logvol_samples.copy()
        for i in range(logvol_samples.shape[0]):
            xvExp[i] = np.exp(logvol_samples[i]) / np.mean(np.exp(logvol_samples[i]))

        if (flagErr):
            moms_model = self.ModelMoments_M(H=H, lambda2=lambda2, K1=K1, T=T, LagSignal=lagSignal, flagLambda2=flagLambda2)
        else:
            moms_model = np.zeros(len(lagSignal))

        lagMax = int(np.max(lagSignal))

        ### Let us padd xm with mean values
        xmz = np.zeros(logvol_samples.shape[1])
        xm = np.zeros((logvol_samples.shape[0], 2 * logvol_samples.shape[1]))
        for i in range(xm.shape[0]):
            xm[i, :] = np.append(xvExp[i, :], xmz)

        xm_size = logvol_samples.shape[1]

        ### We dont want to subtract theoretical error

        ##### Computing the value of correlation at lag 0
        lagMax1 = lagMax
        # lagMax1 = 0
        # m0 = (xm[:,0:xm_size-lagMax1]*xm[:,0:xm_size-lagMax1]).mean(axis=1)

        m0 = np.ones(2)

        m1 = (xm[:, 0:xm_size - lagMax1] * xm[:, lagSignal[0]:lagSignal[0] + xm_size - lagMax1]).mean(axis=0) / m0.mean(
            axis=0) - moms_model[0]

        m2 = (xm[:, 0:xm_size - lagMax1] * xm[:, lagSignal[1]:lagSignal[1] + xm_size - lagMax1]).mean(axis=0) / m0.mean(
            axis=0) - moms_model[1]
        mm = np.vstack((m1, m2))
        for i in range(2, len(lagSignal)):
            m3 = (xm[:, 0:xm_size - lagMax1] * xm[:, lagSignal[i]:lagSignal[i] + xm_size - lagMax1]).mean(axis=0) / m0.mean(
                axis=0) - moms_model[i]
            mm = np.vstack((mm, m3))
        return np.array(mm)

    def criterion(self,params, *args):
        '''
        --------------------------------------------------------------------
        This function computes the GMM weighted sum of squared moment errors
        criterion function value given parameter values and an estimate of
        the weighting matrix.
        --------------------------------------------------------------------
        INPUTS:
        params = (4,) vector, ([H, lambda2,T,lsigma2])

        args   = length 3 tuple, (xvals, lagSig,W_hat)
        xvals  = (N,) vector, values of log-variance
        lagSig = array of lags
        W_hat  = (R, R) matrix, estimate of optimal weighting matrix


        RETURNS: crit_val
        --------------------------------------------------------------------
        '''
        H, lambda2, T = params
        samples, lagSig, lsigma2, W, flagLambda2 = args

        err = self.ErrorVecSignal(samples, H, lambda2, T, lsigma2, lagSig, flagLambda2=flagLambda2)
        err = err.mean(axis=1)

        crit_val = np.dot(np.dot(err.T, W), err)

        return crit_val

    def criterion_M(self,params, *args):
        '''
        --------------------------------------------------------------------
        This function computes the GMM weighted sum of squared moment errors
        criterion function value given parameter values and an estimate of
        the weighting matrix.
        --------------------------------------------------------------------
        INPUTS:
        params = (4,) vector, ([H, lambda2,T,lsigma2])

        args   = length 3 tuple, (xvals, lagSig,W_hat)
        xvals  = (N,) vector, values of log-variance
        lagSig = array of lags
        W_hat  = (R, R) matrix, estimate of optimal weighting matrix


        RETURNS: crit_val
        --------------------------------------------------------------------
        '''
        H, lambda2, K1 = params
        samples, T, lagSig, W, flagLambda2 = args

        err = self.ErrorVecSignal_M(samples, H, lambda2, K1, T, lagSig, flagLambda2=flagLambda2)
        err = err.mean(axis=1)

        crit_val = np.dot(np.dot(err.T, W), err)

        return crit_val

    def criterion_m(self,params, *args):
        '''
        --------------------------------------------------------------------
        This function computes the GMM weighted sum of squared moment errors
        criterion function value given parameter values and an estimate of
        the weighting matrix.
        --------------------------------------------------------------------
        INPUTS:
        params = (4,) vector, ([H, lambda2,T,lsigma2])

        args   = length 3 tuple, (xvals, lagSig,W_hat)
        xvals  = (N,) vector, values of log-variance
        qvals   = vector, qvalues to match moments
        lagSig = array of lags
        W_hat  = (R, R) matrix, estimate of optimal weighting matrix


        RETURNS: crit_val
        --------------------------------------------------------------------
        '''
        H, lambda2, lsigma2 = params
        samples, lagSig, qvals, W, flagLambda2 = args

        err = self.ErrorVecSignal_m(samples, H, lambda2, lsigma2, lagSig, qvals, flagLambda2=flagLambda2)
        err = err.mean(axis=1)

        crit_val = np.dot(np.dot(err.T, W), err)

        return crit_val

    def ComputeParamsGMM(self,datastream_sample, LagSignal=np.array([1, 2, 4, 5, 8, 11, 16, 22, 32, 45, 64, 90, 128]), niter=5,
                         GMM_Method=1, qvals=None, flagPlot=True,
                         flagRemoveLowFreq=False, flagLambda2=True, method='L-BFGS-B', H_Init=-1, lambda2_Init=-1):

        '''
        --------------------------------------------------------------------
        Main GMM procedure
        --------------------------------------------------------------------
        INPUTS:
            xvals: vector corresponbding to the sample of the S-fBm (or the log of the volatility)
            lagSig: list of lags used for the GMM method
            niter : number of iterations in the GMM procedure (niter = 5 by default)
            GMM_Method: type of Moments one uses: 0: Correlation of the measure M = exp(xvals) at lags lagSig
                                                  1: Covariances of xvals at lags lagSig (Default value)
                                                  2: Moments of exp(xvals) at scales lagSig

            flagRemoveLowFreq : Boolean, removes low frequencies (second order polynomial fit) on xvals if True
            flagLambda2 : Boolean, estimates lambda^2 if True, nu^2 if False
            method : Optimisation method of GMM procedure. Possible choices are : "L-BFGS-B" (default),"Powell",
                                                           "TNC","BFGS","Nelder-Mead","SLSQP".




        RETURNS:
           HH_gmm,l2_gmm, np.exp(T_gmm), ls2_gmm, pvalue, J, J_95 with:
               HH_gmm : Estimated Hurst index value
               l2_gmm : intermittency coefficient lambda^2 (of variance nu^2) estimated value
               np.exp(T_gmm): integral scale estimated value
               ls2_gmm : log-noise variance value
               pvalue, J, J_95 : values for testing as in  "Anine E. Bolko, Kim Christensen,
               Mikko S. Pakkanen, Bezirgen Veliyev , A GMM approach to estimate the roughness of stochastic volatility",
               https://arxiv.org/pdf/2010.04610.pdf
        --------------------------------------------------------------------
        '''

        ### For J-test
        J = 0

        ### Case we have a single array
        if (len(datastream_sample.shape) == 1):
            datastream_sample = np.array([datastream_sample])

        if (GMM_Method == 2):
            qvals = 2

        ### Parameters init
        if (H_Init < 0):
            H, lambda2 = self.HurstEstimator(datastream_sample, LagSignal[4:])
        if (lambda2_Init > 0):
            lambda2 = lambda2_Init
        if (H_Init > 0):
            H = H_Init

        if (flagLambda2):
            zl = -1 * np.log(1.0 / np.abs(lambda2) - 1)

        else:
            zl = np.log(lambda2 / np.abs(H) + 0.0001)
        zz = -1 * np.log(1.0 / np.abs(H) - 1)

        ### Initial values of constant
        T0 = np.log(datastream_sample.shape[1] ** (2 * H) / (1 + 2 * H) / (1 + H))
        LL = datastream_sample.shape[1]

        zs = -10
        lsigma2_gmm1 = zs
        #### Removing eventual low frequencies
        if (flagRemoveLowFreq):
            for i in range(datastream_sample.shape[0]):
                xx = np.array(range(len(datastream_sample[i])))
                cc = np.polyfit(xx, datastream_sample[i], deg=2)
                datastream_sample[i] = datastream_sample[i] - cc[0] * xx * xx - cc[1] * xx - cc[2]

        T_gmm1 = T0
        tol = 1e-15
        if (GMM_Method == 1):
            # params_init = np.array([H,lambda2,T0,lsigma2])
            params_init = np.array([zz, zl, T0])
            if (flagLambda2):
                bounds = ((None, None), (None, None), (-5, 10))

            else:
                bounds = ((None, None), (-5, 10), (-10, 8))

            W_hat = np.eye(len(LagSignal))

            gmm_args = (datastream_sample, LagSignal, lsigma2_gmm1, W_hat, flagLambda2)
            res = opt.minimize(self.criterion, params_init, args=(gmm_args), method=method, tol=tol, bounds=bounds)

            H_gmm1, lambda2_gmm1, T_gmm1 = res.x

        if (GMM_Method == 0):
            params_init = np.array([zz, zl, 0])
            if (flagLambda2):
                bounds = ((None, None), (None, None), (-10, 20))

            else:
                bounds = ((-30, -1e-2), (-5, 10), (-10, 20))
                bounds = ((None, None), (-5, 10), (-10, 20))

            W_hat = np.eye(len(LagSignal))

            gmm_args = (datastream_sample, np.log(LL), LagSignal, W_hat, flagLambda2)
            res = opt.minimize(self.criterion_M, params_init, args=(gmm_args), method=method, tol=tol, bounds=bounds)
            # res = opt.minimize(criterion_M,params_init,args=(gmm_args),method=method,tol=tol)
            H_gmm1, lambda2_gmm1, T_gmm1 = res.x

        if (GMM_Method == 2):
            # params_init = np.array([H,lambda2,lsigma2])
            params_init = np.array([zz, zl, zs])
            if (flagLambda2):
                # =((-10,-1e-2),(-10,-1e-2),(-10,10))
                bounds = ((None, None), (None, None), (-10, 10))
            else:
                bounds = [(None, None), (-10, 5), (-10, 5)]

            W_hat = np.eye(len(LagSignal))
            gmm_args = (datastream_sample, LagSignal, qvals, W_hat, flagLambda2)
            res = opt.minimize(self.criterion_m, params_init, args=(gmm_args), method=method, tol=tol, bounds=bounds)
            # res = opt.dual_annealing(criterion_m,x0=params_init,args=(gmm_args),tol=tol,method=method)
            H_gmm1, lambda2_gmm1, lsigma2_gmm1 = res.x

        ### iteration in order to adjust the weighting matrix
        for i in range(niter):

            if (GMM_Method == 0):
                err1 = self.ErrorVecSignal_M(datastream_sample, H_gmm1, lambda2_gmm1, T_gmm1, np.log(LL), LagSignal, flagLambda2)

            if (GMM_Method == 1):
                err1 = self.ErrorVecSignal(datastream_sample, H_gmm1, lambda2_gmm1, T_gmm1, lsigma2_gmm1, LagSignal, flagLambda2)

            if (GMM_Method == 2):
                err1 = self.ErrorVecSignal_m(datastream_sample, H_gmm1, lambda2_gmm1, lsigma2_gmm1, LagSignal, qvals, flagLambda2)

            ## WW = err1.dot(err1.T)/err1.shape[1]

            #### Compute Hac Estimation (Newey-West Kernel)
            err2 = np.subtract(err1.T, err1.mean(axis=1))
            err2 = err2.T
            WW = NeweyWest_CovarianceMatrixEstimation(err2)
            try:
                W_hat = lin.pinv(WW)
            except:
                W_hat = np.eye(WW.shape[0], WW.shape[1])

            if (GMM_Method == 0):
                # gmm_args = (xvals,T_gmm1,lagSig,W_hat,flagLambda2)
                gmm_args = (datastream_sample, np.log(LL), LagSignal, W_hat, flagLambda2)
                params = np.array([H_gmm1, lambda2_gmm1, T_gmm1])
                # params = np.array([H_gmm1,lambda2_gmm1,lsigma2_gmm1])
                res = opt.minimize(self.criterion_M, params, args=(gmm_args), method=method, tol=tol, bounds=bounds)
                # res = opt.minimize(criterion_M,params_init,args=(gmm_args),method=method,tol=tol)

                H_gmm1, lambda2_gmm1, T_gmm1 = res.x

                # J = criterion(np.array([H_gmm1,lambda2_gmm1,lsigma2_gmm1]),xvals,T_gmm1,lagSig,W_hat,flagLambda2)
                J = self.criterion_M(np.array([H_gmm1, lambda2_gmm1, T_gmm1]), datastream_sample, np.log(LL), LagSignal, W_hat, flagLambda2)
                J *= len(datastream_sample[0])

            if (GMM_Method == 1):
                # gmm_args = (xvals,T_gmm1,lagSig,W_hat,flagLambda2)
                gmm_args = (datastream_sample, LagSignal, lsigma2_gmm1, W_hat, flagLambda2)
                params = np.array([H_gmm1, lambda2_gmm1, T_gmm1])
                # params = np.array([H_gmm1,lambda2_gmm1,lsigma2_gmm1])
                res = opt.minimize(self.criterion, params, args=(gmm_args), method=method, tol=tol, bounds=bounds)
                # res = opt.minimize(criterion,params_init,args=(gmm_args),method=method,tol=tol)

                H_gmm1, lambda2_gmm1, T_gmm1 = res.x


                # J = criterion(np.array([H_gmm1,lambda2_gmm1,lsigma2_gmm1]),xvals,T_gmm1,lagSig,W_hat,flagLambda2)
                J = self.criterion(np.array([H_gmm1, lambda2_gmm1, T_gmm1]), datastream_sample, LagSignal, lsigma2_gmm1, W_hat, flagLambda2)
                J *= len(datastream_sample[0])

            if (GMM_Method == 2):
                gmm_args = (datastream_sample, LagSignal, qvals, W_hat, flagLambda2)
                params = np.array([H_gmm1, lambda2_gmm1, lsigma2_gmm1])
                res = opt.minimize(self.criterion_m, params, args=(gmm_args), method=method, tol=tol, bounds=bounds)
                # res = opt.minimize(criterion_m,params_init,args=(gmm_args),method=method,tol=tol)
                H_gmm1, lambda2_gmm1, lsigma2_gmm1 = res.x
                J = self.criterion_m(np.array([H_gmm1, lambda2_gmm1, lsigma2_gmm1]), datastream_sample, LagSignal, qvals, W_hat,
                                flagLambda2)
                J *= len(datastream_sample[0])

        print("Objective function value after calibration = ",res.fun)
        HH_gmm1 = 1 / (1 + np.exp(-H_gmm1))
        if (flagLambda2):
            l2_gmm1 = 1 / (1 + np.exp(-lambda2_gmm1))
        else:
            l2_gmm1 = np.exp(lambda2_gmm1)

        # ls2_gmm1 = 2/(1+np.exp(-lsigma2_gmm1))
        ls2_gmm1 = np.exp(lsigma2_gmm1)

        if (GMM_Method == 0) or (GMM_Method == 2):
            pvalue = chi2.sf(J, len(LagSignal) - 2)
            J_95 = chi2.isf(0.05, len(LagSignal) - 2)

        if (GMM_Method == 1):
            pvalue = chi2.sf(J, len(LagSignal) - 3)
            J_95 = chi2.isf(0.05, len(LagSignal) - 3)

        if (flagPlot):
            plt.figure()

            if (GMM_Method == 1):
                ss = LagSignal
                ss1 = ss
                if (ss[0] == 0):
                    ss1 = ss[1:]
                err1 = self.ErrorVecSignal(datastream_sample, H_gmm1, lambda2_gmm1, T_gmm1, lsigma2_gmm1, ss, flagErr=False, flagLambda2=flagLambda2)
                cc1 = err1.mean(axis=1)
                ### Model Theoretical Covariance
                mm = self.ModelMoments(H_gmm1, lambda2_gmm1, T_gmm1, lsigma2_gmm1, LagSignal=ss, flagLambda2=flagLambda2)
                plt.plot(ss, cc1, 'o-',color = 'red',label='Empirical mean error GMM')
                plt.plot(ss, mm,label='Model moments GMM')
                plt.legend()
                m_th = mm.copy()
                m_emp = cc1.copy()
                ss_l = ss.copy()

                mm =self.ModelMoments(H_gmm1, lambda2_gmm1, T_gmm1, lsigma2_gmm1, LagSignal=ss1, flagLambda2=flagLambda2)
                err2 = self.ErrorVecSignal(datastream_sample, H_gmm1, lambda2_gmm1, T_gmm1, lsigma2_gmm1, ss1, flagErr=False,
                                  flagLambda2=flagLambda2)
                cc2 = err2.mean(axis=1)
                plt.figure()
                plt.plot(np.log(ss1), cc2, 'o-',color = 'green',label='Empirical mean error GMM')
                plt.plot(np.log(ss1), mm,label='Model moments GMM')
                plt.legend()
            if (GMM_Method == 0):
                ss = LagSignal
                ss1 = ss
                if (ss[0] == 0):
                    ss1 = ss[1:]

                err1 = self.ErrorVecSignal_M(datastream_sample, H_gmm1, lambda2_gmm1, T_gmm1, np.log(LL), ss, flagErr=False,flagLambda2=flagLambda2)
                cc1 = err1.mean(axis=1)
                ### Model Theoretical Covariance
                mm = self.ModelMoments_M(H_gmm1, lambda2_gmm1, T_gmm1, np.log(LL), LagSignal=ss, flagLambda2=flagLambda2)
                plt.plot(ss, cc1, 'o-',label='Empirical mean error GMM')
                plt.plot(ss, mm,label='Model moments GMM')
                plt.legend()
                m_th = mm.copy()
                m_emp = cc1.copy()
                ss_l = ss.copy()

                mm = self.ModelMoments_M(H_gmm1, lambda2_gmm1, T_gmm1, np.log(LL), LagSignal=ss1, flagLambda2=flagLambda2)
                err2 = self.ErrorVecSignal_M(datastream_sample, H_gmm1, lambda2_gmm1, T_gmm1, np.log(LL), ss1, flagErr=False,flagLambda2=flagLambda2)
                cc2 = err2.mean(axis=1)
                plt.figure()
                plt.plot(np.log(ss1), cc2, 'o-',label='Empirical mean error GMM - log scale')
                plt.plot(np.log(ss1), mm,label='Model moments GMM - log scale')
                plt.legend()

            if (GMM_Method == 2):
                err1 = self.ErrorVecSignal_m(datastream_sample, H_gmm1, lambda2_gmm1, lsigma2_gmm1, LagSignal, qvals, flagErr=False,flagLambda2=flagLambda2).mean(axis=1)
                dd = np.zeros((1, len(LagSignal)))
                xx = np.zeros((1, len(LagSignal)))
                for i in range(1):
                    dd[i, :] = err1[i * len(LagSignal):(i + 1) * len(LagSignal)]
                    xx[i, :] = self.ModelMoments_m(H_gmm1, lambda2_gmm1, lsigma2_gmm1, LagSignal=LagSignal, qval=qvals, flagLambda2=flagLambda2)
                    plt.plot(np.log(LagSignal), (dd[i, :]), 'o',label='Empirical mean error GMM - log scale')
                    plt.plot(np.log(LagSignal), (xx[i, :]), '-',label='Model moments GMM - log scale')
                    plt.legend()
                m_emp = np.ones(1)
                m_th = np.ones(1)
                ss_l = np.ones(1)

            plt.show()

            ### Sargan-Hansen Test
        if (flagLambda2):
            print('H = ', HH_gmm1, 'lambda2 =', (l2_gmm1), 'T = ', np.exp(T_gmm1), 'lsigma2 =', ls2_gmm1, 'p-Value= ',
                  pvalue, 'J = ', J, 'J_95 = ', J_95)
            print('------------------------------------------------------')
        else:
            print('H = ', HH_gmm1, 'nu2 =', (l2_gmm1), 'T = ', np.exp(T_gmm1), 'lsigma2 =', ls2_gmm1, 'p-Value= ',
                  pvalue, 'J = ', J, 'J_95 = ', J_95)
            print('------------------------------------------------------')
        if (flagPlot):
            return HH_gmm1, l2_gmm1, np.exp(T_gmm1), ls2_gmm1, pvalue, J, J_95, [m_emp, m_th, ss_l]
        else:
            return HH_gmm1, l2_gmm1, np.exp(T_gmm1), ls2_gmm1, pvalue, J, J_95

    def MultipleGMMCalibrations(self,datastream_samples, LagSignal=np.array([1, 2, 4, 5, 8, 11, 16, 22, 32, 45, 64, 90, 128]), niter=5,
                         GMM_Method=1, qvals=None, flagPlot=False,
                         flagRemoveLowFreq=False, flagLambda2=True, method='L-BFGS-B', H_Init=-1, lambda2_Init=-1):
        Outputparameters = dict()
        if (flagPlot):
            Outputparameters_columns = ['Assets','H', 'lambda2', 'exp(T)', 'exp(lsigma2)', 'pvalue', 'J', 'J_95','external_parameters']
        else:
            Outputparameters_columns = ['Assets','H', 'lambda2', 'exp(T)', 'exp(lsigma2)', 'pvalue', 'J', 'J_95']
        for column in Outputparameters_columns:
            Outputparameters[column] = []
        for asset,logvolcurve in datastream_samples.items():
            Outputparameters['Assets'] += [asset]
            calibrated_params = self.ComputeParamsGMM(logvolcurve, LagSignal, niter,GMM_Method, qvals, flagPlot,flagRemoveLowFreq, flagLambda2, method, H_Init, lambda2_Init)
            for i in range(len(calibrated_params)):
                Outputparameters[Outputparameters_columns[i+1]] += [calibrated_params[i]]
        return pd.DataFrame.from_dict(Outputparameters)

