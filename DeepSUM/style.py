import tensorflow as tf
import numpy as np
import os
#from vgg19 import *
from tensorflow.keras.models import load_model
        
def autoencoder_layers(layer_names, weights):
        autoencoder_model = load_model(weights)
        autoencoder_model.trainable = False
        
        #autoencoder_model.layers.pop(0)
        #newInput = tf.keras.Input(shape=(96,96,1)) 
        #autoencoder_model.layers.append(newInput)
        #autoencoder_model2 = tf.keras.Model(newInput, newOutputs)
        #autoencoder_model.summary()
        outputs = [autoencoder_model.get_layer(name).output for name in layer_names]

        model = tf.keras.Model([autoencoder_model.input], outputs)
        #inputs = autoencoder_model.layers[0].
        return model

def gram_matrix(input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)

#def gram_matrix(features, normalize = True):
        #batch_size , height, width, filters = features.shape
        #batch_size , height, width, filters = tf.shape(features)##features.shape
        #features = tf.reshape(features, (8,50176, filters))
        #features = tf.reshape(features, (8,50176, filters))
        #features = tf.reshape(features, (filters, height*width, filters))#height*width
        #features = tf.reshape(features,[64, -1])

        #tran_f = tf.transpose(features)
        #gram = tf.matmul(features,tran_f)
        #if normalize:
            ##gram /= tf.cast(height*width, tf.float32)
            #gram /= tf.cast(50176, tf.float32)
            
        #features = tf.reshape(features, (1, -1, 64))#height*width
        #batch_size , height, width, filters = tf.shape(features)
        #tran_f = tf.transpose(features, perm=[0,2,1])
        #gram = tf.matmul(tran_f, features)

        #return gram

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers, weights):
        
        super(StyleContentModel, self).__init__()
        self.vgg = autoencoder_layers(style_layers + content_layers, weights)
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
        

    def call(self, inputs):
        #preprocessed_input = preprocess_input(inputs) ##DJB no need for this function
        
        outputs = self.vgg(inputs) ##24/4/ updated ##not preprocessed
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        #for style_output in style_outputs:
            #print (tf.shape(style_output))
            ##batch_size , height, width, filters = tf.shape(style_output)
            #gram_matrix(style_output)

        # Compute the gram_matrix
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        # Features that extracted by VGG
        style_dict = {style_name:value for style_name, value in zip(self.style_layers, style_outputs)}
        content_dict = {content_name:value for content_name, value in zip(self.content_layers, content_outputs)}

        return {'content':content_dict, 'style':style_dict}