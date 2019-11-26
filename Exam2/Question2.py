import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import SGD,Adam,RMSprop
from keras.utils import np_utils

activation_function="sigmoid"# "softplus"
n_classes=1
n_epoch=20
n_sample=1000
n_batch=64
x_train=np.loadtxt('Q2_Xtrain.txt',delimiter=' ')
x_test=np.loadtxt('Q2_Xtest.txt',delimiter=' ')

#train the model
x_input=x_train[:,0].T
y_input=x_train[:,1].T

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
        model = Sequential([Dense(n_perceptron,input_dim=1,activation=activation_function),
                            Dense(n_classes,input_dim=n_perceptron,activation = None)
                            ])
        model.compile(optimizer = 'adam',loss = 'mean_squared_error',metrics= ['accuracy'])

        #Begin training
        converged=0
        tolerance=0.01
        last_score=0
        while converged==0:
            model.fit(x_train, y_train, epochs = n_epoch, batch_size=n_batch, verbose = 0)
            score = model.evaluate(x_test,y_test)

            if np.abs(score[0]-last_score)<tolerance:
                converged=1
            last_score=score[0]
            print last_score
            print (n_perceptron,k)
        #save the result
        Accuracy[k,n_perceptron-1]=last_score

print Accuracy
# perceptron_mean=np.mean(Accuracy,axis=1)
# print 

