
import os
import numpy as np 
import tensorflow as tf 
import scipy.io
import matplotlib.pyplot as plt

style_path="style_1"
content_path="content_1"
style=plt.imread(style_path+".jpg")
content=plt.imread(content_path+'.jpg')

# plt.axis('off')
# plt.title('content image')
# plt.imshow(content)
# plt.show()

# plt.axis('off')
# plt.title('style image')
# plt.imshow(style)
# plt.show()

style_weight = 1
content_weight = 0.5
style_layer = ['relu1_2','relu2_2','relu3_2']
content_layer = ['relu1_2']
print content.shape

def stylize(style_image,content_image,learning_rate=0.1,epochs=500):
	target = tf.Variable(tf.random_normal(content_image.shape),dtype=tf.float32)
	style_input = tf.constant(style_image,dtype=tf.float32)
	content_input = tf.constant(content_image,dtype=tf.float32)
	cost = loss_function(style_weight,content_weight,style_input,content_input,target)
	optimization = tf.train.AdamOptimizer(learning_rate).minimize(cost)
	with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
		tf.initialize_all_variables().run()
		for i in range(epochs):
			_.loss,target_image = sess.run([optimization,cost,target])
			print("iter: %d, loss: %.9f" %(i,loss))
			if (i+1)%100 == 0:
				image = np.clip(target_image+128,0,255).astype(np.uint8)
				image.fromarray(image).save("output/%s.jpg" % content_path)

def conv_relu(w_b,input):
	conv = tf.nn.conv2d(input, w_b[0], strides=[1, 1, 1, 1], padding='SAME')
	relu = tf.nn.relu(conv+wb[1])
	return relu

def get_wb(layers, i):
    #w = tf.constant(layers[i][2][0][0][0])
    #bias = layers[i][2][0][0][0]
    print  layers[i].shape
    # b = tf.constant(np.reshape(bias, (bias.size)))
    w=1
    b=1
    return w, b
def pool(input):
 	return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def loss_function(style_weight,content_weight,style_image,content_image,target_image):
	style_feature = vgg19([style_image])
	content_feature = vgg19([content_image])
	target_feature = vgg19([target_image])

	loss = 0
	for layer in content_layer:
		loss += content_weight * content_loss(target_feature,content_feature)
	for layer in style_layer:
		loss += style_weight * style_loss(target_feature,style_feature)
	return loss

def content_loss(target_feature,content_feature):
	_,height,width,channel = map(lambda i:i.value, content_feature.get_shape())
	content_size = height * width * channel
	return tf.nn.l2_loss(target_feature - content_feature) / content_size

def style_loss(target_feature,style_feature):
	_,height,width,channel = map(lambda i:i.value, style_features.get_shape())
	style_size = height * width * channel
	target_feature = tf.reshape(target_feature,(-1,channel))
	target_gram = tf.matmul(tf.transpose(target_feature),target_feature)/style_size

	style_feature = tf.reshape(style_feature,(-1, channel))
	style_gram = tf.matmul(tf.transpose(style_feature),style_feature)/style_size
	return tf.nn.l2_loss(target_gram - style_gram) / style_size

def vgg_parameter():
	vgg_params = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
	return vgg_params
	
vgg_params = vgg_parameter()
vgg_layers = vgg_params['layers'][0]

neural_net = {}
neural_net['input'] = tf.Variable(np.zeros([1, content.shape[0], content.shape[1], 3]), dtype=tf.float32)
neural_net['conv1'] = conv_relu(neural_net['input'], get_wb(vgg_layers, 0))
neural_net['conv2'] = conv_relu(neural_net['conv1'], get_wb(vgg_layers, 2))
neural_net['pool1'] = pool(neural_net['conv2'])
neural_net['conv3'] = conv_relu(neural_net['pool1'], get_wb(vgg_layers, 5))
neural_net['conv4'] = conv_relu(neural_net['conv3'], get_wb(vgg_layers, 7))
neural_net['pool2'] = spool(neural_net['conv4'])


