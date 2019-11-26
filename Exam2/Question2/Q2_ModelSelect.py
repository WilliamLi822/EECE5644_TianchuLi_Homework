import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import SGD,Adam,RMSprop
from keras.utils import np_utils


x_train=np.loadtxt('Xtrain.txt',delimiter=' ')
x_test=np.loadtxt('Xtest.txt',delimiter=' ')

sigmoid_output=np.loadtxt('sigmoid_MSE.txt',delimiter=' ')
softplus_output=np.loadtxt('softplus_MSE.txt',delimiter=' ')

sigmoid_mean=np.mean(sigmoid_output,axis=0)
softplus_mean=np.mean(softplus_output,axis=0)
plt.plot(np.arange(1,11), sigmoid_mean,'r')
plt.plot(np.arange(1,11), softplus_mean,'g')
plt.title('Model Select')
plt.xlabel('Number of Perceptrons')
plt.ylabel('Mean Squared Error')
plt.legend(['Sigmoid', 'Softplus'])
plt.show()

if np.amin(sigmoid_mean)>np.amin(softplus_mean):
	activation_function="softplus"
	n_perceptron=np.argmin(softplus_mean)+1
else:
	activation_function="sigmoid"
	n_perceptron=np.argmin(sigmoid_mean)+1

print "the number of perceptron of best perform model: ",n_perceptron
print "the activation function of best perform model: ",activation_function

model = Sequential([Dense(n_perceptron,input_dim=1,activation=activation_function),
                    Dense(1,input_dim=n_perceptron,activation = None)
                    ])
model.compile(optimizer='adam',loss='mean_squared_error')

#Begin training with best perform model
converged=0
tolerance=0.001
last_score=0
while converged==0:
    model.fit(x_train[:,0], x_train[:,1], epochs = 20, batch_size=64, verbose = 0)
    score = model.evaluate(x_test[:,0], x_test[:,1])
    print np.abs(score-last_score)
    print score
    if np.abs(score-last_score)<tolerance:
        converged=1
    last_score=score

print "MSE of test dataset with best model: ",last_score
y_predict=model.predict(x_test[:,0])
plt.subplot(121)
plt.plot(x_test[:,0], y_predict,'.r')
plt.title('Predict Distribution')
plt.xlabel('x1 value')
plt.ylabel('predict x2 value')
plt.subplot(122)
plt.plot(x_test[:,0], x_test[:,1],'.r')
plt.title('True Distribution')
plt.xlabel('x1 value')
plt.ylabel('x2 value')
plt.show()


