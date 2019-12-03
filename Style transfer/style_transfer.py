import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

model_path="./vgg19.npy"
data_dict=np.load(model_path,encoding='latin1').item()
def print_layer(t):
    print t.op.name, ' ', t.get_shape().as_list(), '\n'


style_path="style_1.jpg"
content_path="content_1.jpg"
style=plt.imread(style_path)
content=plt.imread(content_path)

def conv(x,d_out,name,fineturn=False, xavier=False):
	d_in=x.shape
	with tf.name_scope(name) as scope:
		if fineturen:
			kernel=tf.constant(data_dict[name][0],name="weights")
			bias=tf.constant(data_dict[name][0],name="bias")
		elif not xavier:
			kernel=tf.Variable(tf.truncated_normal([3,3,d_in,d_out], stddev=0.1), name='weights')



	conv = tf.nn.conv2d(x, kernel,strides=[1, 1, 1, 1], padding='SAME')
	#strides=[batch, height, width, channels], the step of move in different dimensions
	activation=tf.nn.relu(conv+bias,name=scope)
	print_layer(activation)
	return activation

def maxpool(x,name):
	activation=tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1, 2, 2, 1], padding='SAME', name=name)
    # ksize=[batch, height, weight, channels], the size of pooling windows
    # trides=[batch, height, width, channels], the step of move in different dimensions
    print_layer(activation)
	return activation

