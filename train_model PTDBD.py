#!pip install numpy==1.16.2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

ecgs = np.load("morelowpass.npy",allow_pickle=True)

#X_train = ecgs[0:100:1,0]
X = list()
Y = list()
np.random.shuffle(ecgs)

for ecg in ecgs:
    for beat in ecg[0]:
        X.append(beat) 
        Y.append(ecg[1][0])
        #individual.append(ecg[2][0])
pass
X = np.asarray(X)
Y = np.asarray(Y)



X_train = X[0:round(len(X)*7/10):1]
X_val = X[(round(len(X)*7/10))+1:round(len(X)*9/10):1]
X_pred = X[(round(len(X)*9/10))+1:len(X):1]

Y_train = Y[0:round(len(Y)*7/10):1]
Y_val = Y[round(len(Y)*7/10)+1:round(len(Y)*9/10):1]
Y_pred = Y[(round(len(Y)*9/10))+1:len(Y):1]

X_train = X_train.reshape(len(X_train),600,12,1)
X_val = X_val.reshape(len(X_val),600,12,1)
X_pred = X_pred.reshape(len(X_pred),600,12,1)

Y_train = to_categorical(Y_train)
Y_val = to_categorical(Y_val)
#Y_pred = to_categorical(Y_pred)

#create Keras model
model = Sequential()
#add some layers to model

model.add(Conv2D(50, kernel_size=(10,2), activation='relu', input_shape=(600,12,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(50,kernel_size=(10,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(50,kernel_size=(4,2),activation='relu'))

model.add(Flatten())
model.add(Dense(2, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[sensitivity,specificity])
plot_model(model, to_file='model.png', show_shapes=True)
#train the model
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=1,verbose=1,shuffle=True)
a = model.predict_classes(X_pred).astype(int)

Y_pred = Y_pred.astype(int)
cm1 = confusion_matrix(a, Y_pred)
sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )
specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)
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