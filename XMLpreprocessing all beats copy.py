from ECGXMLReader import ECGXMLReader as XMLread
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import os 
import readchar

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

#Returns the peaks and the leads
def getPeaksnEcg(inp):
    ecg = XMLread(inp)
    leads = ecg.getAllVoltages()
    #fig, ax_list = plt.subplots(4, 2,sharex='all')
    #ax_list = ax_list.flatten()
    b = list()  
    idx=0
    for k,v in leads.items():
        #sig = butter_highpass_filter(v, 10, 1500, 5)/100
            if v.shape == (600,):
                v = signal.resample(v, 300)
            sig = butter_lowpass_filter(butter_highpass_filter(np.asarray(v,dtype=float), 20, 1500, 5), 50, 1000, 5)/100
            a = np.asarray(sig)
            b.append(a/2)
            #sig = butter_lowpass_filter(np.asarray(v), 5, 1000, 5)
            #ax_list[idx].plot(sig,linewidth=0.5)
            #ax_list[idx].set_ylabel(k)
            idx=idx+1
           
    a = np.asarray(b).T
    d = a[:,1]
    c = np.convolve(np.square(np.gradient(d,1)),np.ones(6))  
    
    
    
    refractory_period = 100   # to have a QRS after less than 200 ms is physiologically impossible
    threshold = max(c)/2    #Threshold should be one 1/3 of the maximum peak in registration
    # Pan-Tompkins continues
    peaks = list()
    for idx, val in enumerate(c):
        # Unpythonic ik
        refractory_period+=1
        if idx - 1 > 0 and idx + 1 < len(c) and c[idx - 1] < val and refractory_period>100 and c[idx + 1] < val and val > threshold :
            #plt.axvline(x=idx,linewidth=1,color = 'k')
            refractory_period = 0
            peaks.append(idx)

    peaks = np.asarray(peaks)
    return peaks,a

#Returns an array of beats per lead
def getIndividualBeats(inp):
    peaks, ecgs = getPeaksnEcg(inp)
    ecg = list()
    for idx2,lead in enumerate(ecgs.T):
        beats = list()
        for idx, val in enumerate(peaks):
            if idx > 0 and idx < len(peaks) - 1:
                slice = lead[int(val - 50):int(val + 100):1]
                beats.append(slice)
                #ax_list[idx2].plot(slice,linewidth=0.25)

        
        ecg.append(beats)
    
    ecg = np.asarray(ecg)

    #The data is dirty, pan-tomkins is bad. Exclude beats with high MSE
    median = np.median(ecg,1) #Make the median
    for idx,lead in enumerate(median):
        ax_list[idx].plot(lead.T,linewidth=0.8)

    #List for the beats we keep
    ecg_clean = list()
    #Loop over all the 8 lead beats and compare with median
    for idx, beat in enumerate(ecg):
        #mse = np.mean(median - beat)
        mse=0
        
        for idx2, lead in enumerate(beat):  
            if len(beat)>8:
                break
            A = median[idx2]
            B = lead
            mse = np.mean((median - lead)**2)
            if  mse > 0.1:
                print(idx + idx2,' is ',mse)
                break
        if mse < 0.2:
            ecg_clean.append(beat)
                #ax_list[0].plot(mean[0].T,linewidth=0.25)

    #Supposed to return individual beats organized per lead
    return np.array(ecg_clean)


#MAIN
LVH = list()
ecgs = list()

for root, dirs, files in os.walk("LEECH"):
        
       
        for ind,f in enumerate(files):
            fig, ax_list = plt.subplots(4, 2,sharex='all')
            ax_list = ax_list.flatten()
            test = getIndividualBeats(os.path.join(root, f))
            short = f.rstrip('_.xml')
            #print(short) 
            #print(short[-1:])
            #LVH.append(short[-1:])
            # making two lists one with LVH 1 or 0 and one with the complete ecg
            LVH.append(int(short[-1:]))
            ecgs.append(test)
            for idx,beat in enumerate(test):
                ax_list[idx].plot(beat.T,linewidth=0.2)
            #key = readchar.readkey()
            #plt.waitforbuttonpress()
            #print(idx)
            plt.text(0,0,f)
            plt.show()


#plt.subplots_adjust(left=0.10,right=0.90,bottom=0.10,top=0.90)

pass
pass