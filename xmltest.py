from ECGXMLReader import ECGXMLReader as XMLread
import matplotlib.pyplot as plt
import numpy as np

ecg = XMLread('033_LEECH_24_0_.xml')
leads = ecg.getAllVoltages()
fig, ax_list = plt.subplots(4, 2,sharex='all')
ax_list = ax_list.flatten()

idx=0
for k,v in leads.items():
        ax_list[idx].plot(v,linewidth=0.5)
        ax_list[idx].set_ylabel(k)
        idx=idx+1
plt.subplots_adjust(left=0.10,right=0.90,bottom=0.10,top=0.90)
plt.show()
pass
pass