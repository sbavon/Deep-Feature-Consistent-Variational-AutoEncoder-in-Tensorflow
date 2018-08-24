
import numpy as np
import tensorflow as tf
from vgg16 import vgg16

class dfc_vae_model(object):
    
    def __init__(self, shape, inputs, alpha = 1, beta = 0.5, vgg_layers = [], learning_rate = 0.0005):
        self.shape = shape
        self.img_input = inputs
        self.alpha = alpha
        self.beta = beta
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.vgg_layers = vgg_layers
        self.learning_rate = learning_rate
    
    def _get_weights(self, name, shape):
        with tf.variable_scope("weights", reuse=tf.AUTO_REUSE) as scope:
            w = tf.get_variable(name=name + '_W',
                                shape=shape,
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        return w
    
    def _get_biases(self, name, shape):
        with tf.variable_scope("biases", reuse=tf.AUTO_REUSE) as scope:
            b = tf.get_variable(name=name + '_b',
                                shape=shape,
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        return b
    
    def _conv2d_bn_relu(self, inputs, name, kernel_size, in_channel, out_channel, stride, activation=True,bn=True):
        with tf.variable_scope(name) as scope:
            
            ### setup weights and biases
            filters = self._get_weights(name, shape=[kernel_size, kernel_size, in_channel, out_channel])
            biases = self._get_biases(name, shape=[out_channel])
            
            ### convolutional neural network
            conv2d = tf.nn.conv2d(input=inputs,
                                 filter=filters,
                                 strides=[1,stride,stride,1],
                                 padding='SAME',
                                 name=name + '_conv')
            conv2d = tf.nn.bias_add(conv2d, biases, name=name+'_add')
            
            ### in case of batch normalization
            if bn == True:
                conv2d = tf.contrib.layers.batch_norm(conv2d, 
                                              center=True, scale=True, 
                                              is_training=True,
                                              scope='bn')
            
            ### in case of leaky relu activation
            if activation == True:
                conv2d = tf.nn.leaky_relu(conv2d, alpha=0.1, name=name)
        
        return conv2d
        
    def encoder(self, reuse=False):
        
        with tf.variable_scope("encoder", reuse = reuse):
            ### Conv2d_bn_relu Layer 1
            conv1 = self._conv2d_bn_relu(self.img_input,
                                   name="conv1",
                                   kernel_size=4,
                                   in_channel=3,
                                   out_channel=32,
                                   stride=2)

            ### Conv2d_bn_relu Layer 2
            conv2 = self._conv2d_bn_relu(conv1,
                                   name="conv2",
                                   kernel_size=4,
                                   in_channel=32,
                                   out_channel=64,
                                   stride=2)

            ### Conv2d_bn_relu Layer 3
            conv3 = self._conv2d_bn_relu(conv2,
                                   name="conv3",
                                   kernel_size=4,
                                   in_channel=64,
                                   out_channel=128,
                                   stride=2)

            ### Conv2d_bn_relu Layer 4
            conv4 = self._conv2d_bn_relu(conv3,
                                   name="conv4",
                                   kernel_size=4,
                                   in_channel=128,
                                   out_channel=256,
                                   stride=2)

            ### flatten the output
            conv4_flat = tf.reshape(conv4, [-1, 256*4*4])

            ### FC Layer for mean
            fcmean = tf.layers.dense(inputs=conv4_flat,
                                  units=100,
                                 activation=None,
                                 name="fcmean")

            ### FC Layer for standard deviation
            fcstd = tf.layers.dense(inputs=conv4_flat,
                                   units=100,
                                   activation=None,
                                   name="fcstd")
        
        ### fcmean and fcstd will be used for sample z value (latent variables)
        return fcmean, fcstd + 1e-6
        
    def decoder(self,inputs, reuse=False):
        
        with tf.variable_scope("decoder", reuse = reuse):
            ### FC Layer for z
            fc = tf.layers.dense(inputs=inputs,
                                units = 4096,
                                activation = None)
            fc = tf.reshape(fc, [-1, 4, 4, 256])

            ### Layer 1
            deconv1 = tf.image.resize_nearest_neighbor(fc, size=(8,8))
            deconv1 = self._conv2d_bn_relu(deconv1,
                                   name="deconv1",
                                   kernel_size=3,
                                   in_channel=256,
                                   out_channel=128,
                                   stride=1)

            ### Layer 2
            deconv2 = tf.image.resize_nearest_neighbor(deconv1, size=(16,16))
            deconv2 = self._conv2d_bn_relu(deconv2,
                                   name="deconv2",
                                   kernel_size=3,
                                   in_channel=128,
                                   out_channel=64,
                                   stride=1)

            ### Layer 3
            deconv3 = tf.image.resize_nearest_neighbor(deconv2, size=(32,32))
            deconv3 = self._conv2d_bn_relu(deconv3,
                                   name="deconv3",
                                   kernel_size=3,
                                   in_channel=64,
                                   out_channel=32,
                                   stride=1)  

            ### Layer 4
            deconv4 = tf.image.resize_nearest_neighbor(deconv3, size=(64,64))
            deconv4 = self._conv2d_bn_relu(deconv4,
                                           name="deconv4",
                                           kernel_size=3,
                                           in_channel=32,
                                           out_channel=3,
                                           stride=1,
                                           activation=False,
                                           bn=False)
            
        return deconv4
    
    def load_vgg(self):
        
        ### pass the input image to VGG model
        #self.resize_input_img = tf.image.resize_images(self.img_input, [224,224])
        #self.vgg_input = VGG(self.resize_input_img)
        #self.l1_r, self.l2_r, self.l3_r = self.vgg_input.load(reuse=False)
        
        ### pass the generated image to VGG model
        #self.resize_gen_img = tf.image.resize_images(self.gen_img, [224,224])
        #self.vgg_gen = VGG(self.resize_gen_img)
        #self.l1_g, self.l2_g, self.l3_g = self.vgg_gen.load(reuse=True)
        self.resize_input_img = tf.image.resize_images(self.img_input, [224,224])
        self.vgg_real = vgg16(self.resize_input_img, 'vgg16_weights.npz')
        self.l1_r, self.l2_r, self.l3_r = self.vgg_real.get_layers()
        
        self.resize_gen_img = tf.image.resize_images(self.gen_img, [224,224])
        self.vgg_gen = vgg16(self.resize_gen_img, 'vgg16_weights.npz')
        self.l1_g, self.l2_g, self.l3_g = self.vgg_gen.get_layers()
        
    def calculate_loss(self):
        
        ### calculate perception loss
        #l1_loss = (tf.reduce_sum(tf.square(self.l1_r-self.l1_g)))/tf.cast(tf.size(self.l1_r), tf.float32)
        #l2_loss = (tf.reduce_sum(tf.square(self.l2_r-self.l2_g)))/tf.cast(tf.size(self.l2_r), tf.float32)
        #l3_loss = (tf.reduce_sum(tf.square(self.l3_r-self.l3_g)))/tf.cast(tf.size(self.l3_r), tf.float32)
        l1_loss = tf.reduce_sum(tf.square(self.l1_r-self.l1_g), [1,2,3])
        l2_loss = tf.reduce_sum(tf.square(self.l2_r-self.l2_g), [1,2,3])
        l3_loss = tf.reduce_sum(tf.square(self.l3_r-self.l3_g), [1,2,3])
        self.pct_loss = tf.reduce_mean(l1_loss + l2_loss + l3_loss)
        
        ### calculate KL loss
        self.kl_loss = tf.reduce_mean(-0.5*tf.reduce_sum(
            1 + self.std - tf.square(self.mean) - tf.exp(self.std), 1))
        
        ### calculate total loss
        self.loss = tf.add(self.beta*self.pct_loss,self.alpha*self.kl_loss)
        
    def optimize(self):
        
        ### create optimizer
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=self.gstep, var_list=var_list)
    
    def build_model(self,reuse=tf.AUTO_REUSE):
        
        ### get mean and std from encoder
        self.mean, self.std = self.encoder(reuse)
        
        ### sampling z and use reparameterization trick
        epsilon = tf.random_normal((tf.shape(self.mean)[0],100), mean = 0.0, stddev=1.0)
        self.z = self.mean + epsilon * tf.exp(.5*self.std)
        
        ### decode to get a generated image
        self.gen_img = self.decoder(self.z,reuse)
        
        ### load vgg
        self.load_vgg()
        
        ### calculate loss
        self.calculate_loss()
        
        ### setup optimizer
        self.optimize()
        
        ### generate random latent variable for random images
        self.random_latent = tf.random_normal((tf.shape(self.mean)[0], 100))
        self.ran_img = self.decoder(self.random_latent,reuse)
        
    ### load VGG weight
    def load_vgg_weight(self, weight_file, sess):
        self.vgg_real.load_weights(weight_file,sess)
        self.vgg_gen.load_weights(weight_file,sess)
        
                
