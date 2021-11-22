
import timeit
start = timeit.default_timer()
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.layers import Dense, LSTM, GRU, Input
from keras.layers import Dropout
from keras.layers import Flatten,  TimeDistributed, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Model
from keras.layers import concatenate
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering( 'th' )
import warnings
warnings.filterwarnings("ignore")
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
import timeit
start = timeit.default_timer()
####################################base-model#############
def baseline_model1():
    model_in = Input(shape=(16, 1, 300))
    l_1=(Convolution2D(64, 1, 1, activation= 'relu' ))(model_in)   
    l_2=(Convolution2D(64, 1, 1, activation= 'relu' ))(l_1) 
    l_3=(Convolution2D(64, 1, 1, activation= 'relu' ))(l_2)
    l_4=(Convolution2D(64, 1, 1, activation= 'relu' ))(l_3)
    l_5=(Convolution2D(64, 1, 1, activation= 'relu' ))(l_4)
    l_6=(Flatten())(l_5)
    model_out = Dense(64, activation='relu')(l_6)
    return model_in, model_out 

m11,m12 = baseline_model1()

def baseline_model2():
    model_in = Input(shape=(16, 300))
    out1=(LSTM(64, activation= 'relu', return_sequences=True))(model_in) 
    out2=(LSTM(64, activation= 'relu',name ="e" ))(out1)
    model_out=Dense(64, activation= 'relu' )(out2)
    return model_in, model_out 
m21, m22= baseline_model2()
c1 = concatenate([m12, m22])
out1 = (Dense(64, activation='relu' ))(c1)
c2=(Reshape((-1,1,128), input_shape=(-1,1,128)))(c1)
c3= (Reshape((-1,128), input_shape=(-1,128)))(c1)

####################################################after_fusion_concatenation###########

def baseline_model3():
    model_in = c2
    l_11=(Convolution2D(64, 1, 1, activation= 'relu' ))(model_in) 
    l_21=(Convolution2D(64, 1, 1, activation= 'relu' ))(l_11) 
    l_31=(Convolution2D(64, 1, 1, activation= 'relu' ))(l_21)
    l_41=(Convolution2D(64, 1, 1, activation= 'relu' ))(l_31)
    l_51=(Convolution2D(64, 1, 1, activation= 'relu' ))(l_41)
    l_61=(Flatten())(l_51)
    model_out = Dense(64, activation='relu')(l_61)
    return model_in, model_out 
m31,m32 = baseline_model3()

def baseline_model4():
    model_in = c3
    o1=(LSTM(64, activation= 'relu', return_sequences=True))(model_in) 
    o2=(LSTM(64, activation= 'relu',name ="e1" ))(o1)
    model_out=Dense(64, activation= 'relu' )(o2)
    return model_in, model_out 
m41,m42 = baseline_model4()

concatenated = concatenate([m32, m42])
o11 = (Dense(64, activation='relu' ))(concatenated)
o22 = (Dense(64, activation='relu'))(o11)
o33 = (Dense(32, activation='relu'))(o22)
o44 = (Dense(8, activation='softmax'))(o33)
merged_model1 = Model([m11, m21], o44) 
merged_model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

       
INPUT_SIGNAL_TYPES = [
        "AX1",
        "AY1",
        "AZ1",
        "GX1",
        "GY1",
        "GZ1",
        "MX1",
        "MY1",
        "MZ1",
        "LX1",
        "LY1",
        "LZ1",
        "OW1",
        "OX1",
        "OY1",
        "OZ1"                
]
# Output classes to learn how to classify
LABELS = [
    "Still", 
    "Walk", 
    "Run", 
    "Bike", 
    "Bus",
    "Car",
    "Train",
    "Subway"
]

TRAIN = "train/new_300/"
TEST = "test/new_300/"

def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        X_signals.append(
                [np.array(serie, dtype=np.float32) for serie in [
                        row.replace('  ', ' ').strip().split(' ') for row in file
                        ]]
                )
        file.close()
    return np.transpose(np.array(X_signals), (1, 2, 0))

X_train_signals_paths = [
         TRAIN + signal + ".txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
         TEST + signal + ".txt" for signal in INPUT_SIGNAL_TYPES
]

X_train = load_X(X_train_signals_paths)
print(X_train.shape)
X_test = load_X(X_test_signals_paths)

print(X_train.shape)
print(X_test.shape)
      

# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
            [elem for elem in [
                    row.replace('  ', ' ').strip().split(' ') for row in file
                    ]], 
                    dtype=np.int32
    )
    file.close()

# Substract 1 to each output class for friendly 0-based indexing 
    return y_ - 1
y_train_path = TRAIN + "y_train.txt"
y_test_path =  TEST + "y_test.txt"

y_train = load_y(y_train_path)
print(y_train.size)
y_test = load_y(y_test_path)

X_train1 = X_train.reshape(X_train.shape[0],16, 1, 300).astype( 'float32' )
X_test1 = X_test.reshape(X_test.shape[0], 16, 1, 300).astype( 'float32' )
print(X_train1.shape)
print(X_test1.shape)

X_train2 = X_train.reshape(X_train.shape[0],16, 300).astype( 'float32' )
X_test2 = X_test.reshape(X_test.shape[0], 16, 300).astype( 'float32' )
print(X_train2.shape)
print(X_test2.shape)        

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# Fit the model
def modelf2():
	m2=merged_model1.fit([X_train1, X_train2], y_train, nb_epoch=1, batch_size=100, verbose=2, shuffle=True)
	return m2

def modele2():
	e2=merged_model1.evaluate([X_test1,X_test2],y_test, verbose=0)
	return e2

def modelp2():
	p2=merged_model1.predict([X_test1, X_test2])
	return p2
merged_model1.fit([X_train1, X_train2], y_train, nb_epoch=1, batch_size=100, verbose=2, shuffle=True)
scores = merged_model1.evaluate([X_test1,X_test2],y_test, verbose=0)
acc=scores[1]

print(" final acc::" + str(acc))    
stop = timeit.default_timer()
    
print('Time: ', stop - start)
