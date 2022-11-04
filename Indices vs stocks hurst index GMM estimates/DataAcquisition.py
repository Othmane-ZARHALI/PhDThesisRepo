#  Paper:   "From Rough to Multifractal volatility: the log S-fBM model"
#  Purpose: Data acquisition of the market data

# Author : Jean-Francois MUZY, Othmane ZARHALI


# Importations
import numpy as np
import pandas as pd
from scipy.signal import correlate
import yfinance as yf
from pandas_datareader import data
from scipy.interpolate import interp1d
import pickle
from ExternalParameters import *



### Correlation function between tauMin =dMin and tauMax = dMax (can be negative)
    ### Returns the xvalues and the correlation value
def Correlation(signal1, signal2, dMin=0, dMax=50, flag_FFT=False, flagCenter=True):
    if (flagCenter):
        signal1 = signal1 - signal1.mean()
        signal2 = signal2 - signal2.mean()
    if flag_FFT:
        correlation = correlate(signal1, signal2, mode='full', method='fft')
    else:
        correlation = correlate(signal1, signal2, mode='full', method='direct')
    correlation = correlation / len(signal1)
    return np.array(range(dMin, dMax + 1)), correlation[range(len(signal1) - 1 + dMin, len(signal1) + dMax)]


def NeweyWest_CovarianceMatrixEstimation(err1):
    ### Err1 is of shape (N_moments, L) where L is the sample size
    ## size of NW windows
    L = err1.shape[1]
    J = int(L ** 0.2)
    s1 = err1.shape[0]
    WW = np.zeros((s1, s1))
    for i in range(-J, J + 1):
        SJ = np.zeros((s1, s1))
        m1 = max(0, i)
        m2 = min(L, L + i)
        for k in range(m1, m2):
            SJ += np.outer(err1[:, k], err1[:, k - i]) / L
        WW += (1 - np.abs(i) / (J + 1)) * SJ
    return WW



class DataAcquisition:
    def __init__(self, source,Filepath=""):
        if type(source)!= str:
            TypeError("DataAcquisition error: source is not of expected type, str")
        else:
            if source=="Yahoo finance" and Filepath !="":
                ValueError("DataAcquisition error: Filepath should be empty for Yahoo finance data")
        if type(Filepath) != str:
            TypeError("DataAcquisition error: Filepath is not of expected type, str")
        else:
            self.source = source
            self.Filepath = Filepath
            self.indices_list = []
            self.market_capitalization = []
            self.dataframe_indices = dict()
            if source =="Yahoo finance":
                self.Marketdata_over_assets = None
                self.logvolcolumnexists = None

    def IndicesCharging(self,symbol="",first_date="",last_date=""):  # symbol can be a list of indices or a single index
        if self.source=="OxfordManInstitute":
            pd_indices = pd.read_csv(self.Filepath)
            # mask = (pd_indices.Symbol == '.STI') | (pd_indices.Symbol == '.KSE')
            mask = (pd_indices.Symbol == '.SPX') | (pd_indices.Symbol == '.STOXX50E')
            pd_indices = pd_indices.drop(index=pd_indices[mask].index)
            pd_indices.reset_index(inplace=True)
            self.indices_list = list(pd_indices.Symbol.unique())
            self.dataframe_indices = pd_indices
        elif self.source=="Yahoo finance":
            self.dataframe_indices = yf.download(symbol, start=first_date, end=last_date)
            self.market_capitalization = data.get_quote_yahoo(symbol)['marketCap'].values

    ### removes Zeros by linear interpolation
    def removeZeros(self,signal):
        #x = np.arange(len(signal))
        # idx = np.nonzero(yy)
        idx = np.where(signal > 0)[0]
        cleaned_signal = signal[idx[0]:]
        x = np.arange(len(cleaned_signal))
        #idx = np.where(cleaned_signal > 0)[0]
        # yy = yy[idx[0]:]
        # idx = np.nonzero(yy)
        if (len(idx) == len(cleaned_signal)):
            return cleaned_signal
        else:
            f = interp1d(x[idx], cleaned_signal[idx])
            return f(x)

    def ComputeLogVolEstimator(self, estimation_type='GK',signal=pd.DataFrame()):
        if signal.empty:
            df = self.dataframe_indices
        else:
            df = signal
        if estimation_type == 'OC':
            logvol = np.log(df.Open / df.Close)
            logvol = np.abs(logvol)
            # ll = removeZeros(ll)
            # ll = np.log(ll)
        elif estimation_type == 'HL':
            logvol = np.log(df.High / df.Low)
            # ll = ll.values
            # ll = removeZeros(ll)
            # ll = np.log(ll)
        elif estimation_type == 'YZ':
            oo = df.Open[1:]
            cc = df.Close[1:]
            cm1 = df.Close[:-1]
            hh = df.High[1:]
            lo = df.Low[1:]
            logvol = (np.log(oo / cm1)) ** 2 + 0.5 * (np.log(hh / lo)) ** 2 - 2 * (np.log(2.0) - 1) * (np.log(cc / oo)) ** 2
            # ll = np.log(ll)
        else:  #https://www.cmegroup.com/trading/fx/files/a_estimation_of_security_price.pdf  GARMAN-KLASS vol estimator
            oo = np.maximum(df.Open, df.Low)
            logvol = 0.5 * (np.log(df.High / df.Low)) ** 2 - (2 * np.log(2.0) - 1) * (np.log(df.Close / oo)) ** 2
            # ll = removeZeros(ll)
        logvol = np.log(logvol)
        logvol.replace([np.inf, -np.inf], np.nan, inplace=True)
        return logvol.interpolate(limit_direction='both').values

    def GetCompleteData_withlogvol(self,symbole_list,first_date='2000-01-03',last_date="2034-01-01"):
        complete_data = []
        for i in range(len(symbole_list)):
            #data =   #ryfdata(symlist[i], fdate=fdate)
            self.IndicesCharging(symbole_list[i], first_date, last_date)
            data = self.dataframe_indices
            data = data.dropna()
            if (data.index[0] == pd.Timestamp(first_date)):
                complete_data.append(data)
        ### Computing the date range
        date = complete_data[0].index
        for i in range(1, len(complete_data)):
            date = date.append(complete_data[i].index)
            date = date.unique()
        ### Synchronization phase
        for i in range(0, len(complete_data)):
            ddd = complete_data[i].index
            z = date.difference(ddd)
            nn = len(z)
            for k in range(nn):
                z1 = z[k]
                ww1 = np.where(ddd > z1)[0]
                if (len(ww1) > 0):
                    i2 = ww1[0]
                else:
                    i2 = -1
                ww2 = np.where(ddd < z1)[0]
                if (len(ww2) > 0):
                    i1 = ww2[-1]
                else:
                    i1 = 0
                rr = 0.5 * (complete_data[i].iloc[i1] + complete_data[i].iloc[i2])
                complete_data[i].loc[z1] = rr
            complete_data[i].sort_index(inplace=True)
            ### Adding a logVolField
        for i in range(len(complete_data)):
            # log = self.ComputeLogVolEstimator(complete_data[i], est_type='HL')
            # complete_data[i]['logVol'] = lv
            complete_data[i]['logVol'] =  self.ComputeLogVolEstimator('GK',complete_data[i])  #HL estimator at the beginning
        #Data synchronization
        self.logvolcolumnexists,self.Marketdata_over_assets = True,complete_data
        return complete_data

    # def getSyncData(self,fdate='2000-01-03', field='logVol'):
    #     lll = []
    #     #mm = ryfdata(symlist[i])
    #     #mm = self.dataframe_indices[symlist[i]]
    #     if (mm.index[0] <= pd.Timestamp(fdate)):
    #         mask = mm.index > pd.Timestamp(fdate)
    #         mm = mm[mask]
    #         if (field == 'logVol'):
    #             ll = self.ComputeLogVolEstimator('HL')  #lvEstimator(mm, est_type='HL')
    #             lll.append(ll)
    #     return lll

    ## Compute the minimum period common to all assets (min year = 2000)
    def ComputeBestList(self):
        filtered_assets = []
        if self.indices_list == [] :
            self.IndicesCharging()
        for i in range(len(self.indices_list)):
            mask = self.dataframe_indices.Symbol == self.indices_list[i]
            # fdd = pd.to_datetime(self.dataframe_indices[mask][self.dataframe_indices[mask].columns[0]].values[0])
            fdd = pd.to_datetime(self.dataframe_indices[mask][self.dataframe_indices[mask].columns[1]].values[0])
            if (fdd.year == 2000):
                filtered_assets.append(self.indices_list[i])
        return filtered_assets

    ## Get the log-volatility of some given asset
    def GetlogVol(self,financial_index, voltype='bv', flagLog=True, flagRemoveZeros=True, flagG=False, y1=2000, y2=2021):
        assets_list_g = self.ComputeBestList()
        index = assets_list_g.index(financial_index)
        if flagG:
            assetName = assets_list_g[index]
        else:
            assetName = self.indices_list[index]
        mask = self.dataframe_indices.Symbol == assetName
        vols = np.abs(self.dataframe_indices[mask][voltype].values)
        if y1!=0 and y2!=0:
            #target_assets = self.dataframe_indices[mask][self.dataframe_indices[mask].columns[0]].values
            target_dates = self.dataframe_indices[mask][self.dataframe_indices[mask].columns[1]].values
            dates = np.array([pd.to_datetime(target_dates[i]).year for i in range(len(vols))])
            ii = (dates > y1) & (dates < y2)
            vols = np.abs(self.dataframe_indices[mask][voltype].values[ii])
            #return vols
        if flagRemoveZeros:
            for k in np.where(vols == 0)[0]:
                vols[k] = 0.5 * (vols[k - 1] + vols[k + 1])
        if flagLog:
            return np.log(vols)
        else:
            return vols

    #### Computation of log-Variance over a window
    def GetlogVolVar(self,logvolSignal, Size):
        vol_over_period = np.zeros(len(logvolSignal) - Size)
        for j in range(0, len(logvolSignal) - Size):
            s1 = logvolSignal[j:j + Size]
            vol_over_period[j] = s1.var()
        return vol_over_period.mean()

    def GetlogVolVar_vs_Size(self,logvolSignal, nn=32):
        scaleMax = len(logvolSignal)
        wSigSize = np.geomspace(8, scaleMax, nn)
        wSigSize = wSigSize.astype('int32')
        overall_vol = np.zeros(nn)
        for i in range(len(overall_vol)):
            overall_vol[i] = self.GetlogVolVar(logvolSignal, wSigSize[i])
        return wSigSize, overall_vol

    ###### Computing the mean and variance values of log each day of the week
    def ComputeMeanVarianceinWeek(self, financial_index, voltype='bv'):
        index = self.indices_list.index(financial_index)
        assetName = self.indices_list[index]
        mask = self.dataframe_indices.Symbol == assetName
        #vols = np.abs(self.dataframe_indices[mask][voltype].values)
        #day_of_the_week = np.array([pd.to_datetime(vols[i]).day_of_week for i in range(len(vols))])
        target_dates = self.dataframe_indices[mask][self.dataframe_indices[mask].columns[1]].values
        #day_of_the_week = np.array([pd.to_datetime(vols[i]).weekday() for i in range(len(vols))])
        day_of_the_week = np.array([pd.to_datetime(target_dates[i]).weekday() for i in range(len(target_dates))])
        log_vols = self.GetlogVol(self.indices_list[index], voltype)
        mm = log_vols.mean()
        m1,m2,m3,m4 = np.zeros(5),np.zeros(5),np.zeros(5),np.zeros(5)
        for i in range(0, 5):
            #days_index = (day_of_the_week == i)
            # m1[i] = log_vols[days_index].mean()

            m1[i] = np.array([log_vols[i] for j in range(len(day_of_the_week)) if day_of_the_week[j]==i]).mean()
            #m2[i] = np.mean(np.square(log_vols[days_index] - mm))
            m2[i] = np.mean(np.square( np.array([log_vols[i] for j in range(len(day_of_the_week)) if day_of_the_week[j]==i]) - mm))
            # vv1 = log_vols[1:]
            # vv0 = log_vols[:-1]
            # jjj = days_index[:-1]
            # m3[i] = np.mean((vv0[jjj] - mm) * (vv1[jjj] - mm))
            # m4[i] = np.mean((vv0[jjj] - mm) * (vv1[jjj] - mm))
        return m1, m2 #, m3

    def ComputeCovarianceMatrixLogvol(self,symbol_list,first_date='2000-01-03'):
        if not self.logvolcolumnexists:
            signals = self.GetCompleteData_withlogvol(symbol_list,first_date)
        else:
            signals = self.Marketdata_over_assets
        length = len(signals)
        zz = np.zeros((length, length))
        zz_b = np.zeros((length, length))
        for i in range(length):
            for k in range(i, length):
                vv1 = signals[i].logVol.values
                vv1 = vv1 - vv1.mean()
                vv2 = signals[k].logVol.values
                vv2 = vv2 - vv2.mean()
                cc = np.cov(vv1, vv2)
                zz[i, k] = cc[0, 1]
                zz[k, i] = cc[0, 1]

                vv1 = np.log(signals[i].Close.values / signals[i].Open.values)
                vv1 = vv1 - vv1.mean()
                vv2 = np.log(signals[k].Close.values / signals[k].Open.values)
                vv2 = vv2 - vv2.mean()
                cc = np.cov(vv1, vv2)
                zz_b[i, k] = cc[0, 1]
                zz_b[k, i] = cc[0, 1]
        ee1 = np.linalg.eig(zz)
        eeb = np.linalg.eig(zz_b)
        l1 = []
        lb = []
        for i in range(length):
            ss = np.zeros(len(signals[0].logVol))
            for k in range(len(signals)):
                ss += ee1[1][k][i] * signals[k].logVol.values
            l1.append(ss)
        for i in range(length):
            ss = np.zeros(len(signals[0].logVol))
            for k in range(length):
                ss += eeb[1][k][i] * np.log(signals[k].Close.values / signals[k].Open.values)
            lb.append(ss)
        return zz, l1, lb








