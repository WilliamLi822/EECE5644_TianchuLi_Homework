import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import SGD,Adam
from keras.utils import np_utils

n_classes=4 
n_epoch=20
n_sample=10000
learning_rate=0.001

# Read data and true label
x_train=np.loadtxt('Xtrain_%i_3.txt' %n_sample,delimiter=' ')
l_train_tmp=np.loadtxt('Ltrain_%i_3.txt'%n_sample)
for i in range(n_sample):
  l_train_tmp[i]-=1
l_train=keras.utils.to_categorical(l_train_tmp, num_classes=n_classes)
x_input,y_input= x_train,l_train

k=-1
kf = KFold(n_splits=10)
Accuracy=np.zeros((10,10))
#k-Fold cross-validation
for train_ind, test_ind in kf.split(x_input):
    k+=1
    x_train=x_input[train_ind]
    x_test=x_input[test_ind]
    y_train=y_input[train_ind]
    y_test=y_input[test_ind]
    # optimize the number of perceptron
    for n_perceptron in range(1,11):
        #Set Model
        model = Sequential([Dense(n_perceptron,input_dim=3,activation="softplus"),
                            Dense(n_classes,input_dim=n_perceptron,activation='softmax')
                            ])
        sgd = SGD(lr=learning_rate, momentum=0.8, decay=1e-5, nesterov=True)
        model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

        #Begin training
        converged=0
        tolerance=0.001
        last_score=0
        while converged==0:
            model.fit(x_train, y_train, epochs = n_epoch, verbose = 0)
            score = model.evaluate(x_test,y_test,verbose = 0)
            if np.abs(score[1] - last_score)<=tolerance:
                converged=1
            last_score=score[1]
            print (n_perceptron,k)
        #save the result
        Accuracy[k,n_perceptron-1]=last_score

print Accuracy



