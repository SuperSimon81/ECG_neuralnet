#!pip install numpy==1.16.2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

import os 
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras import backend as K
from keras.models import load_model
import keras
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from vis.utils import utils as utils
from vis.visualization import visualize_saliency
import datetime

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

# def plot_map(grads):
#     fig, axes = plt.subplots(1,2,figsize=(14,5))
#     axes[0].imshow(_img)
#     axes[1].imshow(_img)
#     i = axes[1].imshow(grads,cmap="jet",alpha=0.8)
#     fig.colorbar(i)
#     plt.suptitle("Pr(class={}) = {:5.2f}".format(
#                       classlabel[class_idx],
#                       y_pred[0,class_idx]))
                      
ecgs = np.load("morelowpass.npy",allow_pickle=True)

X_train = list()
Y_train = list()

X_val = list()
Y_val = list()

X_pred = list()
Y_pred = list()

#np.random.shuffle(ecgs)

nrtotrain = round(len(ecgs)*7/10)

for ecg in ecgs[0:nrtotrain:1]:
    for beat in ecg[0]:
        X_train.append(beat) 
        Y_train.append(ecg[1][0])
        
for ecg in ecgs[nrtotrain+1:round(len(ecgs)*9/10):1]:
    for beat in ecg[0]:
        X_val.append(beat) 
        Y_val.append(ecg[1][0])
        
for ecg in ecgs[round(len(ecgs)*9/10)+1:len(ecgs):1]:
    for beat in ecg[0]:
        X_pred.append(beat) 
        Y_pred.append(ecg[1][0])
        
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)

X_val = np.asarray(X_val)
Y_val = np.asarray(Y_val)

X_pred = np.asarray(X_pred)
Y_pred = np.asarray(Y_pred)

X_train = X_train.reshape(len(X_train),600,12,1)
X_val = X_val.reshape(len(X_val),600,12,1)
X_pred = X_pred.reshape(len(X_pred),600,12,1)

Y_train = to_categorical(Y_train)
Y_val = to_categorical(Y_val)
#Y_pred = to_categorical(Y_pred)

model = load_model('09:33:16-Mar052020.h5')
print("Loaded model from disk")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.layers[-1].activation = keras.activations.linear
model = utils.apply_modifications(model)

from vis.visualization import visualize_cam
penultimate_layer_idx = utils.find_layer_idx(model, "conv2d_3") 

class_idx  = [0,1]
pred = list()
grad = list()
Y_pred = Y_pred.astype(int)
#xx_pred = X_pred.astype(int) 
a = model.predict_classes(X_pred).astype(int) 
for idx,ecg in enumerate(X_pred):
  
    if((Y_pred[idx] + a[idx]) ==0):
        seed_input = ecg
        #seed_input = X_pred[0]
        grad_top1  = visualize_cam(model, -1, class_idx, seed_input, 
                              penultimate_layer_idx = penultimate_layer_idx,#None,
                               backprop_modifier     = None,
                               grad_modifier         = None)
                            
       
        pred.append(ecg)
        grad.append(grad_top1)
        print(idx)

pred = np.asarray(pred)
grad = np.asarray(grad)

#cm1 = confusion_matrix(a, Y_pred)
#sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
#print('Sensitivity : ', sensitivity1 )
#specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
#print('Specificity : ', specificity1)
#pred = np.mean(pred,0)
grad = np.mean(grad,0)
#case = 15



cmap = pl.cm.jet
my_cmap = cmap(np.arange(cmap.N))
my_cmap[0:64,3] = 0 #np.linspace(0, 1, 64)
from matplotlib.colors import ListedColormap
# Create new colormap
my_cmap = ListedColormap(my_cmap)

#Lead legends
lead_names = np.asarray(['i', 'ii', 'iii', 'aVr', 'aVl', 'aVf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'])
fig, ax_list = plt.subplots(6, 2,sharex='all')
#ax_list = ax_list.flatten()

pred=np.squeeze(pred)
grad = np.transpose(grad)
#for idx,lead in enumerate(pred): #[0:12:1] because we dont want VCG

   #ax_list[idx].plot(np.transpose(lead),linewidth=2,color='black')
   #ax_list[idx].imshow(grad[idx,np.newaxis,:], cmap='jet', aspect="auto",alpha=0.7,extent=[0,600,-2.5,2.5])
   #ax_list[idx].set_ylabel(lead_names[idx])
for ecg in pred:
    for idx,ax in enumerate(ax_list.T.flatten()):
        #ax.plot(ecg[0][:,:,idx].T,linewidth=0.1,alpha=0.1,color='black')
        ax.plot(ecg[:,idx],linewidth=0.1,color='black')

for idx,ax in enumerate(ax_list.T.flatten()):     
    ax.imshow(grad[idx,np.newaxis,:], cmap='jet', aspect="auto",alpha=0.7,extent=[0,600,-2.5,2.5])
    ax.set_ylabel(lead_names[idx])
#grad = grad[0]

#ax_list[idx].plot(grad[idx],linewidth=0.5,color='black') 
#plt.suptitle("(casenr {} predicted class {} class {})".format(case,a[case],Y_pred[case]))

#plt.subplots_adjust(left=0.10,right=0.90,bottom=0.10,top=0.90)
plt.tight_layout()
plt.show()
#plt.savefig()





#print(tn,fp,fn,tp)
#print(sensitivity(a,Y_pred))
#print(specificity(a,Y_pred))

# Plot training & validation accuracy values
#plt.plot(history.history['sensitivity'],color='k')
#plt.plot(history.history['specificity'],color='r')

#plt.plot(history.history['val_sensitivity'],color='k')
#plt.plot(history.history['val_specificity'],color='r')
#plt.title('Model accuracy')
#plt.ylabel('sens/spec')
#plt.xlabel('Epoch')
#plt.legend(['train_sensitivity','train_specificity','val_sensitivity','val_specificity'], loc='upper left')
#plt.show()

#plt.imshow(np.squeeze(X_train[0]))

# fig, ax_list = plt.subplots(6, 2,sharex='all')
# ax_list = ax_list.flatten()
# for ecg in ecgs:
#     pass
#     for idx,lead in enumerate(np.mean(ecg[0].T,2)): #[0:12:1] because we dont want VCG
#         #print(idx)
#         ax_list[idx].plot(lead,linewidth=0.1)
#         #ax_list[idx].axvline(200, linewidth=0.8, color='r')
#         #ax_list[idx].set_ylabel(lead_names[idx])
#         #ax_list[idx].set_autoscaley_on(False)
#         #ax_list[idx].set_autoscalex_on(True)
#         #ax_list[idx].set_ylim([-2, 2])
#         #ax_list[idx].grid(True,'both','both')

#         # ax_list[idx].yaxis.set_major_locator(MultipleLocator(1))
#         # ax_list[idx].yaxis.set_minor_locator(MultipleLocator(0.2))
#         # ax_list[idx].xaxis.set_major_locator(MultipleLocator(200))
#         # ax_list[idx].xaxis.set_minor_locator(MultipleLocator(40))

# plt.subplots_adjust(left=0.10,right=0.90,bottom=0.10,top=0.90)
# plt.show()