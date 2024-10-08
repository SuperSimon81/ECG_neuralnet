from ECGXMLReader import ECGXMLReader as XMLread
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import os 


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
    c = np.convolve(np.square(np.gradient(a[:,1],1)),np.ones(50))  
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
    for lead in ecgs:
        beats = list()
        for idx, val in enumerate(peaks):
            if idx > 0 and idx < len(peaks) - 1:
                slice = lead[int(val - 50):int(val + 100):1]
                beats.append(slice)
            ecg.append(beats)
    
    ecg = np.asarray(ecg)
    #Supposed to return individual beats organized per lead
    return ecg


#MAIN
LVH = list()
ecgs = list()
fig, ax_list = plt.subplots(4, 2,sharex='all')
ax_list = ax_list.flatten()
for root, dirs, files in os.walk("LEECH"):
        for ind,f in enumerate(files):
            test = getIndividualBeats(os.path.join(root, f))
            short = f.rstrip('_.xml')
            #print(short) 
            #print(short[-1:])
            #LVH.append(short[-1:])
            # making two lists one with LVH 1 or 0 and one with the complete ecg
            LVH.append(int(short[-1:]))
            ecgs.append(test)
            #print(idx)

del ecgs[36] #Something is wrong with ecgs[36]



# for root, dirs, files in os.walk("LEECH"):
#     for ind,f in enumerate(files):
#         xml = XMLread(os.path.join(root, f))
#         ecgs = xml.getAllMedianVoltages()
#         count=0
#         print(ecgs['I'].shape)
        
#         for name,ecg in ecgs.items():
#             if ecg.shape == (600,):
#                 sig = signal.resample(ecg, 300)
#             else:
#                 sig = ecg
#             ax_list[count].plot(np.asarray(sig),linewidth=0.5)
#             count=count+1



beats = list()

for idx,ecg in enumerate(ecgs):
    
   for idx2,beat in enumerate(ecg):
               beats.append((beat.T,idx2,LVH[idx]))
               ax_list[idx2].plot(np.asarray(beat.T),linewidth=0.1)
               #print(beat.shape)
#ps,ecg = getPeaks('033_LEECH_24_0_.xml')
#print(shape(ecgs))

beats = np.asarray(beats)
#np.save('LVH.npy',beats,allow_pickle=True)
       

plt.subplots_adjust(left=0.10,right=0.90,bottom=0.10,top=0.90)
plt.show()
pass
pass