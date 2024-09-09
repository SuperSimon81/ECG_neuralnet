#!pip install numpy==1.16.2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os 
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv1D, MaxPooling1D,Flatten, MaxPooling2D,Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.metrics import AUC
from sklearn.metrics import roc_curve, auc

from tensorflow.keras.optimizers import Adam
import tensorflow.keras
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix
#from vis.utils import utils as utils
#from vis.visualization import visualize_saliency
import datetime
import tensorflow as tf
from sklearn.model_selection import KFold, StratifiedKFold
# Import TensorBoard
from tensorflow.keras.callbacks import TensorBoard





def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def save(model):
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%H:%M:%S-%b%d%Y")
    model.save(timestampStr+".h5")
    print("Saved model to disk as "+timestampStr+".h5")

ecgs = np.load("morelowpass.npy",allow_pickle=True)

X = list()
Y = list()

#np.random.seed(1)
#np.random.shuffle(ecgs)

print(len(ecgs))
for ecg in ecgs:
    print(len(ecg[0][:5]))
    for beat in ecg[0][:5]:
        
        X.append(beat) 
        Y.append(ecg[1][0])



print(len(X))
X = np.asarray(X)
Y = np.asarray(Y)

X = X.reshape(len(X),600,12,1)


Y = to_categorical(Y)


# Define Tensorboard as a Keras callback
tensorboard = TensorBoard(
  log_dir='.\logs',
  histogram_freq=1,
  write_images=True
)
keras_callbacks = [
  tensorboard
]


#create Keras model
model = Sequential()
#add some layers to model

model.add(Conv2D(64, kernel_size=(100,3), activation='relu', input_shape=(600,12,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32,kernel_size=(50,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16,kernel_size=(4,2),activation='relu'))

model.add(Flatten())

model.add(Dense(2, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='binary_crossentropy', metrics='AUC') #[sensitivity,specificity]
#plot_model(model, to_file='model.png', show_shapes=True)
plot_model(model,to_file="model1.png")
#model.add(visualkeras.SpacingDummyLayer(spacing=100))
#visualkeras.layered_view(model).show() # display using your system viewer
#layer_idx = -1
#model.layers[layer_idx].activation = keras.activations.linear
#model = utils.apply_modifications(model)

sens_per_fold = []
spec_per_fold = []


batch_size = 25
loss_function = sparse_categorical_crossentropy
no_classes = 2
no_epochs = 1
optimizer = Adam()
verbosity = 1
num_folds = 9
fold_no = 1



#skfold =  StratifiedKFold(n_splits=num_folds,)
kfold = KFold(n_splits=num_folds, shuffle=True)
    
for train, test in kfold.split(X, Y):
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} of {num_folds}...')

    #train the model
    history = model.fit(X[train], Y[train],
                    batch_size=batch_size,
                    epochs=no_epochs,
                    verbose=verbosity)


    scores = model.evaluate(X[test], Y[test], verbose=1)
    #print(f'Score for fold {fold_no}: {model.metrics_names[1]} of {scores[1]}; {model.metrics_names[2]} of {scores[2]}')# {model.metrics_names[2]} of {scores[2]}; {model.metrics_names[3]} of {scores[3]}; {model.metrics_names[4]} of {scores[4]}')
    
    #sens_per_fold.append(scores[1])
    #spec_per_fold.append(scores[2])

    fold_no = fold_no + 1
pass


# == Provide average scores ==
##print('------------------------------------------------------------------------')
##print('Score per fold')
##for i in range(0, len(sens_per_fold)):
##  print('------------------------------------------------------------------------')
#  print(f'> Fold {i+1} - Sens: {sens_per_fold[i]} - Spec: {spec_per_fold[i]}%')
#print('------------------------------------------------------------------------')
#print('Average scores for all folds:')
#print(f'> Sens: {np.mean(sens_per_fold)} (+- {np.std(sens_per_fold)})')
#print(f'> Spec: {np.mean(spec_per_fold)}')
#print('------------------------------------------------------------------------')
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