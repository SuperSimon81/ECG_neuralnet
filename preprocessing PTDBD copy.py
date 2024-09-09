import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os 
from scipy import signal

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


def get_ecg_names():
    data = list()
    infarction = list()
    locale = list()
    for root, dirs, files in os.walk("PTBDB"):
        for file in files:
            if file.endswith(".hea"):
                #print(os.path.join(root, file))
                fp = open(os.path.join(root, file))
                
                diagnosisline = [line for line in fp if line.startswith('# Reason for admission:')]
                diagnosis = diagnosisline[0].rstrip("\n")
                diagnosis = diagnosis[24:]
               
                fp = open(os.path.join(root, file))
                localizationline = [line for line in fp if line.startswith('# Acute infarction (localization):')]
                localization = localizationline[0].rstrip("\n")
                localization = localization[35:]
                
                fname = os.path.join(root, file.rstrip("hea")+"csv")

                if diagnosis == "Myocardial infarction":
                    #print(file.rstrip(".hea")+"csv", "has an infarction")
                    data.append(fname)
                    infarction.append(1)
                  
                    if localization == 'infero-latera':
                        localization = 'infero-lateral'
                    if localization == 'infero-poster-lateral':
                        localization = 'infero-postero-lateral'

                    locale.append(localization)
                    #print(file.rstrip(".hea")+"csv", "has an infarction in: ",localization)
                elif diagnosis == "Healthy control":
                    #print(file.rstrip(".hea") + "csv", "is a healthy control")
                    data.append(fname)
                    infarction.append(0)
                #print(,file.rstrip(".hea")+".csv")

                break
    my_dict = {i:locale.count(i) for i in locale}
    test = dict.fromkeys(locale)



    return data,infarction 

def load_PTBDB_ecg(file):
    # ECG format
    #  Elapsed time','i','ii','iii','avr','avl','avf','v1','v2','v3','v4','v5','v6'
    rawarray = np.loadtxt(file,delimiter=',',skiprows=2,usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
   
    b = list()
    
    for item in rawarray.T:
        ##Highpass to get rid of baseline wander and lowpass to get rid of high frequency noise
        #y = butter_highpass_filter(item, 1, 1500, 5)
        #y = butter_lowpass_filter(item, 25, 1000, 5)
        y = butter_lowpass_filter(butter_highpass_filter(item, 1, 1500, 5), 50, 1000, 5)
        b.append(y)
    a = np.asarray(b).T
    # in principle the line below is Pan-Tompkins from derivative step to moving window step
    c = np.convolve(np.square(np.gradient(a[:,1],1)),np.ones(50))  
    refractory_period = 200   # to have a QRS after less than 200 ms is physiologically impossible
    threshold = max(c)/3    #Threshold should be one 1/3 of the maximum peak in registration
    
    # Pan-Tompkins continues
    peaks = list()
    for idx, val in enumerate(c):
        # Unpythonic
        refractory_period+=1
        if idx - 1 > 0 and idx + 1 < len(c) and c[idx - 1] < val and refractory_period>200 and c[idx + 1] < val and val > threshold :
            # plt.axvline(x=idx,linewidth=0.5,color = 'k')
            refractory_period = 0
            peaks.append(idx)

    peaks = np.asarray(peaks)

    #meanrr = np.mean(np.diff(peaks, 1, 0))

    
    #fig, ax = plt.subplots()
    
    #ax.plot(c*20,linewidth=0.5,color='r')
    #ax.plot(b,linewidth=0.5,color='r')
    #ax.plot(a,linewidth=0.5)
    
    #for num in peaks:
    #    plt.axvline(x=num, linewidth=0.5, color='k')

    beats = list()  # A list to store our individual beats

    #meanrr = np.mean(np.diff(rpeaks, 1, 0))  # Calculate new more accurate RR-interval

    for idx, val in enumerate(peaks):
        if idx > 0 and idx < len(peaks) - 1:
            slice = a[int(val - 200):int(val + 400):1]
            beats.append(slice)
    barr = np.asarray(beats)  # Make into a numpy array for convenience
    #barr = np.take(barr, (0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11), 2)  # reorganize columns for subplots
    #fig, ax = plt.subplots()
    #ax.plot(barr[1])
    
    #plt.show()
    return barr


#lead_names = np.take(lead_names, (0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11))  # Reorganize the columns for subplots

fnames,infarction = get_ecg_names()
ecgs = list()
for idx,f in enumerate(fnames[1:len(fnames)]): #len(fnames)len(fnames)
    ecg = load_PTBDB_ecg(f)
    
    ecgs.append((ecg,np.full(len(ecg),infarction[idx]),np.full(len(ecg),idx),idx))
    print(idx)
    
ecgs = np.asarray(ecgs)

save = False

if save == True:
    np.save('alsoecgnr.npy',ecgs,allow_pickle=True)
    print("saved")

ecgs = np.asarray(ecgs)



plot = True

if plot == True:
    lead_names = np.asarray(['i', 'ii', 'iii', 'aVr', 'aVl', 'aVf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'])
    fig, ax_list = plt.subplots(6, 2,sharex='all')
    #ax_list = ax_list.flatten()
    
    
    for ecg in ecgs:
        for idx,ax in enumerate(ax_list.T.flatten()):  
            
            #print(idx)
            ax.plot(ecg[0][:,:,idx].T,linewidth=0.1,alpha=0.1,color='black')
            ax.set_ylabel(lead_names[idx])
            #ax_list[idx].axvline(200, linewidth=0.8, color='r')
            #ax_list[idx].set_ylabel(lead_names[idx])
            #ax_list[idx].set_autoscaley_on(False)
            #ax_list[idx].set_autoscalex_on(True)
            #ax_list[idx].set_ylim([-2, 2])
            #ax_list[idx].grid(True,'both','both')

            # ax_list[idx].yaxis.set_major_locator(MultipleLocator(1))
            # ax_list[idx].yaxis.set_minor_locator(MultipleLocator(0.2))
            # ax_list[idx].xaxis.set_major_locator(MultipleLocator(200))
            # ax_list[idx].xaxis.set_minor_locator(MultipleLocator(40))

    plt.tight_layout()
    plt.show()
print("done")