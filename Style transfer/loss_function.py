import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def loss_function(style_weight,content_weight,style_image,content_image,target_image,style_layer,content_layer):
	loss = 0.0
	for layer in content_layer:
		loss += content_weight * content_loss



def content_loss(target_feature,content_feature):
	_,height,width,channel = map(lambda i:i.value, content_feature.get_shape())
	content_size = height * width * channel
	return tf.nn.l2_loss(target_feature - content_feature) / content_size

def style_loss(target_features,style_features):
	_,height,width,channel = map(lambda i:i.value, style_features.get_shape())
	style_size = height * width * channel


def gram_matrix(target_feature):
	