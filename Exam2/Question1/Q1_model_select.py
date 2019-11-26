import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import SGD,Adam,RMSprop
from keras.utils import np_utils

n_classes=4 
n_epoch=20
n_sample=1000
learning_rate=0.001

# Read train data and true label
x_train=np.loadtxt('Xtrain_%i_3.txt' %n_sample,delimiter=' ')
l_train_tmp=np.loadtxt('Ltrain_%i_3.txt'%n_sample)
for i in range(n_sample):
  l_train_tmp[i]-=1
l_train=keras.utils.to_categorical(l_train_tmp, num_classes=n_classes)

# Read test data and true label
x_test=np.loadtxt('Xtest.txt',delimiter=' ')
l_test_tmp=np.loadtxt('Ltest.txt')
for i in range(10000):
  l_test_tmp[i]-=1
l_test=keras.utils.to_categorical(l_test_tmp, num_classes=n_classes)

#Read accuracy list with perceptron from 1 to 10
accuracy_list=np.loadtxt('Accuracy_%i.txt' %n_sample,delimiter=' ')

#Find the number of perceptron that achieve maximum accuracy
accuracy_mean=np.mean(accuracy_list,axis=1)
n_perceptron=np.argmax(accuracy_mean)+1
print "\nthe number of perceptron of best perform model: ",n_perceptron
plt.plot(np.arange(1,11), accuracy_mean)
plt.xlabel('Nubmer of Perceptrons')
plt.ylabel('Probability of Correct')
plt.show()

#Set Model with selected parameters
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
    model.fit(x_train, l_train, epochs = n_epoch, verbose = 0)
    score = model.evaluate(x_test,l_test,verbose = 0)
    if np.abs(score[1] - last_score)<=tolerance:
        converged=1
    last_score=score[1]
print "\nProbability of Correct on test dataset with best model: "
print last_score

#Build confusion confusion matrix
l_estimate_tmp=model.predict(x_test)
l_estimate=np.argmax(l_estimate_tmp,axis=1)
confusion=confusion_matrix(l_test_tmp, l_estimate)
true_distribution=np.sum(confusion,axis=0)
Pe=[1.-confusion[0][0]/float(true_distribution[0]),1.-confusion[1][1]/float(true_distribution[1]),
    1.-confusion[2][2]/float(true_distribution[2]),1.-confusion[3][3]/float(true_distribution[3])]
print "\nProbability of error of test dataset: "
print Pe
print "\nConfusion matrixof test dataset: "
print confusion




