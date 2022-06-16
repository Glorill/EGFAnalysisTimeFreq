'''
This software is used for extract group and phase velocity dispersion curves from surface wave empirical Green’s function (EGF) or cross-correlation function (CF) from ambient noise.	
'''

import numpy as np
import pandas as pd
from enum import Enum

from scipy.signal import hilbert, windows
from scipy.fftpack import fft,ifft
from geopy import distance

from matplotlib import pyplot as plt
import seaborn as sns

import logging

log_name = 'EGFAnalysisTimeFreq'
logger = logging.getLogger(log_name)
handler = logging.FileHandler(
    f"{log_name}.log", mode='w', encoding='utf-8')  # overwrite old files
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)   # setting the log level

class gfcn_analysis:
    def __init__(self, FilePath , isEGF = True):
        '''
        FilePath: path of the data file
        isEGF: True for EGF, False for CF
        '''
        self.data_path = FilePath
        try:
            self.RawData = np.loadtxt(FilePath)
        except:
            logger.error(f'Fail to load data from {FilePath}')
            raise

        # station longitude , latitude and altitude
        self.Longitude_A = self.RawData[0, 0]
        self.Latitude_A = self.RawData[0, 1]
        self.Altitude_A = self.RawData[0, 2]
        self.Longitude_B = self.RawData[1, 0]
        self.Latitude_B = self.RawData[1, 1]
        self.Altitude_B = self.RawData[1, 2]
        if self.Longitude_A < 0:
            self.Longitude_A += 360
        if self.Longitude_B < 0:
            self.Longitude_B += 360
        logger.debug(f'Longitude_A: {self.Longitude_A}, Latitude_A: {self.Latitude_A}, Altitude_A: {self.Altitude_A}')
        logger.debug(f'Longitude_B: {self.Longitude_B}, Latitude_B: {self.Latitude_B}, Altitude_B: {self.Altitude_B}')

        # calculate great circle distance
        circleDist = distance.great_circle(
            (self.Latitude_A, self.Longitude_A), (self.Latitude_B, self.Longitude_B)).km
        staElevDiff = abs(self.Altitude_A - self.Altitude_B)/1000
        if np.isnan(staElevDiff):
            staElevDiff = 0
        logger.debug(f'circleDist: {circleDist}, staElevDiff: {staElevDiff}')
        # correct station distance due to elevation difference
        self.StaDist = np.sqrt(circleDist**2 + staElevDiff**2)
        logger.info('Station distance: {} km'.format(self.StaDist))
      
        self.PtNum = self.RawData.shape[0] - 2
        self.Time = self.RawData[2:, 0]
        self.Green_AB = self.RawData[2:, 1]
        self.Green_BA = self.RawData[2:, 2]

        maxamp = max(max(self.Green_AB), max(self.Green_BA))
        if maxamp > 0:
            self.Green_AB /= maxamp
            self.Green_BA /= maxamp

        # using hilbert tranform to obtain EGF from CF if reading CF
        if isEGF == False:
            self.Green_AB = np.imag(hilbert(self.Green_AB))
            self.Green_BA = np.imag(hilbert(self.Green_BA))

        self.SampleT = self.Time[1] - self.Time[0]
        self.SampleF = 1 / self.SampleT

    class GreenFcnObjectsType(Enum):
        A_to_B = 1,
        B_to_A = 2,
        A_add_B = 3

    class VelocityFunType(Enum):
        GroupVelocity = 1,
        PhaseVelocity = 2

    def PhaseGroupVImg(self,
                       StartT=5, EndT=50, DeltaT=0.1,
                       StartV=2, EndV=5.2, DeltaV=0.002,
                       WinAlpha=0.1, NoiseTime=150, MinSNR=5.0,
                       GreenFcnObjectsType=GreenFcnObjectsType.A_add_B,
                       VelocityFunType=VelocityFunType.GroupVelocity,
                       isPlot=True):
        '''
        calculate phase velocity dispersion curve
        args:
            StartT: start time of the analysis
            EndT: end time of the analysis
            DeltaT: time interval of the analysis
            StartV: start velocity of the analysis
            EndV: end velocity of the analysis
            DeltaV: velocity interval of the analysis
            WinAlpha: the proportion of cosine part to the whole window
            NoiseTime: the time of noise sampling
            GreenFcnObjectsType: the type of Green’s function objects
            VelocityFunType: the type of velocity function
            isPlot: True for ploting the result
        return:

        '''
        self.StartT = StartT
        self.EndT = EndT
        self.DeltaT = DeltaT
        self.StartV = StartV
        self.EndV = EndV
        self.DeltaV = DeltaV
        self.WinAlpha = WinAlpha

        # the number of time points
        NumCtrT = int((self.EndT - self.StartT) / self.DeltaT) + 1
        TPoint = np.linspace(self.StartT, self.EndT, NumCtrT)

        # the number of velocity points
        NumCtrV = int((self.EndV - self.StartV) / self.DeltaV) + 1
        VPoint = np.linspace(self.EndV, self.StartV, NumCtrV)
        
        # calculate the typical value of the time difference from the typical value of the velocity as the width of the window function
        self.StartWin = round(self.SampleT * self.StaDist / self.EndV)
        self.EndWin = round(self.SampleT * self.StaDist / self.StartV)
        if self.EndWin > self.PtNum:
            self.EndWin = self.PtNum - 1
            self.StartV = np.ceil(10 * self.StaDist/self.Time[-1])/10
            logger.warning(f'Min velocity reset to {self.StartV}')

        # select function object
        if GreenFcnObjectsType == GreenFcnObjectsType.A_to_B:
            self.GreenFcn = self.Green_AB
        elif GreenFcnObjectsType == GreenFcnObjectsType.B_to_A:
            self.GreenFcn = self.Green_BA
        elif GreenFcnObjectsType == GreenFcnObjectsType.A_add_B:
            self.GreenFcn = (self.Green_AB + self.Green_BA) / 2.0
        
        # calculate the window function
        Window, TaperLen = self.GenerateSignalWindow(
            self.StartWin, self.EndWin, self.PtNum, self.WinAlpha)
        WinWave = self.GreenFcn * Window

        # extract noise window after the windowed surface wave
        NoisePt = round(NoiseTime/self.SampleT)
        NoiseStartIndex = self.EndWin + TaperLen
        if (NoiseStartIndex + NoisePt) < self.PtNum:
            NoiseWinWave = self.GreenFcn[NoiseStartIndex:NoiseStartIndex + NoisePt]
        else:
            NoiseWinWave = self.GreenFcn[NoiseStartIndex:]
            logger.warning(f'Noise window length of {NoiseWinWave.shape[0]}, not long enough')

        WaveShowPt = min((self.EndWin + TaperLen), self.PtNum)
        WinWaveShow = WinWave[:WaveShowPt]

        # calculate envelope images for signal and noise and estimate SNR
        # SNR(T) =  max(signal envelope at period T)/mean(noise envelope at period T)
        EnvelopeImageSignal = self.EnvelopeImageCalculation(
            WinWaveShow, self.SampleF, TPoint, self.StaDist)
        AmpS_T = np.max(EnvelopeImageSignal, axis=1)
        EnvelopeImageNoise = self.EnvelopeImageCalculation(
            NoiseWinWave * windows.tukey(NoisePt, 0.2), self.SampleF, TPoint, self.StaDist)
        SNR_T = AmpS_T / np.mean(EnvelopeImageNoise, axis=1)

        # calculate the velocity
        if VelocityFunType == VelocityFunType.GroupVelocity:
            TravPtV = self.StaDist / (np.asarray(range(self.StartWin - 1, self.EndWin)) * self.SampleT)
            VImg = []
            for i in range(NumCtrT):
                VImg.append(np.interp(
                    VPoint, TravPtV[::-1], (EnvelopeImageSignal[i, self.StartWin:self.EndWin+1]/AmpS_T[i])[::-1]))
            VImg=np.transpose(VImg)
        elif VelocityFunType == VelocityFunType.PhaseVelocity:
            pass

        # packaged data
        self.VImgData = pd.DataFrame(VImg)
        self.VImgData.columns = np.round(TPoint)
        self.VImgData.index = np.round(VPoint, 2)

        # plot the result
        if isPlot:
            fig = plt.figure(figsize=(10, 6))
            fig.subplots_adjust(hspace=0.5, wspace=0.5)

            # plot signal (blue) and noise (red)
            ax = fig.add_subplot(321)
            ax.plot(self.Time[:WaveShowPt], WinWaveShow, 'b-', label='Windowed Signal')
            ax.plot(self.Time[NoiseStartIndex:NoiseStartIndex+NoisePt],
                    NoiseWinWave, 'r-', label='Noise')
            ax.set_xlabel('Time(s)')
            ax.set_ylabel('Amplitude')
            ax.legend()

            # plot SNR
            ax = fig.add_subplot(322)
            ax.set_yscale('log')
            ax.plot(TPoint, SNR_T, 'b-', label='SNR')
            ax.set_xlabel('Period(s)')
            ax.set_ylabel('SNR')
            ax.grid(which='both')
            # flag values greater than MinSNR
            HighSNRIndex = np.where(SNR_T > MinSNR)
            ax.plot(TPoint[HighSNRIndex], SNR_T[HighSNRIndex],
                    'r*', label='SNR > ' + str(MinSNR))
            ax.legend()

            # plot velocity image
            fig.add_subplot(3,1,(2,3))
            # ax = sns.heatmap(df, cmap="RdBu_r")
            # ax = sns.heatmap(df, cmap="Spectral_r")
            ax = sns.heatmap(self.VImgData, cmap="RdYlBu_r")

            ax.set_xlabel('Period(s)')
            ax.set_ylabel('Group Velocity(km/s)')
            plt.show()

        return self.VImgData

    @staticmethod
    def GenerateSignalWindow(StartWin, EndWin, PtNum, Alpha=0.1):
        '''
        generate window function, scaling to length of PtNum

        args:
            StartWin: start index for the value of 1 in the window function
            EndWin: end index for the value of 1 in the window function
            PtNum: length of the original data
            Alpha: the proportion of cosine part to the whole window

        return:
            Window: window function
            TaperLen: the width of one side of the cosine function
        '''
        # window length
        win_len = int((EndWin - StartWin)/(1-Alpha)) + 1
        # generate window function
        Window = windows.tukey(win_len, Alpha)
        TaperLen = round(win_len * Alpha / 2)
        # crop or add the left side
        pad_left_len = StartWin - TaperLen
        if pad_left_len > 0:
            Window = np.pad(Window, (pad_left_len, 0), 'constant')
        else:
            Window = Window[-pad_left_len:]
        # crop or add the right side
        if Window.shape[0] < PtNum:
            Window = np.pad(
                Window, (0, PtNum - Window.shape[0]), 'constant')
        else:
            Window = Window[:PtNum]
        
        return Window, TaperLen

    @staticmethod
    def EnvelopeImageCalculation(WinWave, fs, TPoint, StaDist):
        '''
        calculate envelope image, i.e., to obtain envelope at each T
        new code for group velocity analysis using frequency domain Gaussian filter
        '''

        # linear interpolation
        alfa_x = [0,100,250,500,1000,2000,4000,20000]
        alfa_y = [5, 8, 12, 20, 25, 35, 50, 75]
        guassalfa = np.interp(StaDist, alfa_x, alfa_y)
        logger.info(f'guassalfa: {guassalfa}')

        NumCtrT = TPoint.shape[0]
        PtNum = WinWave.shape[0]

        nfft = int (2 ** int(np.log2(max(PtNum, 1024*fs))))
        xxfft = fft(WinWave, nfft)
        fxx = np.asarray(range(nfft // 2 + 1)) / float(nfft) * fs

        EnvelopeImage = np.zeros((NumCtrT, PtNum))
        for i in range(NumCtrT):
            CtrT = TPoint[i]
            fc = 1/CtrT
            Hf = np.exp(-guassalfa*(fxx - fc) ** 2 / fc ** 2)
            yyfft = xxfft[:nfft // 2 + 1] * Hf
            yyfft = np.append(yyfft, np.conj(yyfft[-2:0:-1]))

            yy = np.real(ifft(yyfft, nfft))
            filtwave = abs(hilbert(yy))
            EnvelopeImage[i,:] = filtwave[0:PtNum]
        return EnvelopeImage

if __name__ == '__main__':
    FilePath = "CF.dat"
    gfcn = gfcn_analysis(FilePath, isEGF=False)
    gfcn.PhaseGroupVImg(DeltaT=0.1,
                        GreenFcnObjectsType=gfcn_analysis.GreenFcnObjectsType.A_add_B)
