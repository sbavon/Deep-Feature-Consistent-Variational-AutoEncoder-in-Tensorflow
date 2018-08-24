
import dfc_vae_model as dfc
import tensorflow as tf
import numpy as np
import util
import matplotlib.pyplot as plt
import os
import sys
import imageio

############# Hyper-Parameters ################
### Adjust parameters in this part
BATCH_SIZE = 32
NUM_EPOCH = 10
VGG_LAYERS = ['conv1_1','conv2_1','conv3_1']
ALPHA = 1
BETA = 8e-6
LEARNING_RATE = 0.0001
IMG_HEIGHT = 64
IMG_WIDTH = 64
TRAINING_DATA = 'celeb_data_tfrecord'
IMG_MEAN = np.array([134.10714722, 102.52040863, 87.15436554])
IMG_STDDEV = np.sqrt(np.array([3941.30175781, 2856.94287109, 2519.35791016]))
###############################################

### restore checkpoint from "Checkpoint" folder
def _restore_checkpoint(saver, sess):
    
    ckpt_path = os.path.dirname(os.path.join(os.getcwd(),'checkpoint/'))
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("get checkpoint")
    return ckpt_path

### create dataset and return iterator and dataset
def _get_data(training_data_tfrecord, batch_size):
    
    dataset = util.read_tfrecord(training_data_tfrecord)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    return iterator, dataset
    
def train_dfc_vae():
    
    ### setup hyper-parameter
    batch_size = BATCH_SIZE
    epoch = NUM_EPOCH
    vgg_layers = VGG_LAYERS
    alpha = ALPHA
    beta = BETA
    learning_rate = LEARNING_RATE
    img_height = IMG_HEIGHT
    img_width = IMG_WIDTH
    training_data_tfrecord = TRAINING_DATA
    
    ### get training data 
    iterator, _ = _get_data(training_data_tfrecord, batch_size)

    ### create iterator's initializer for training data 
    iterator_init = iterator.initializer
    data = iterator.get_next()
    
    ### define input data
    img_input = (tf.reshape(data, shape=[-1, img_height, img_width, 3])-IMG_MEAN) / IMG_STDDEV
    
    ### build model graph
    model = dfc.dfc_vae_model([img_height,img_width], img_input, alpha, beta, vgg_layers, learning_rate)
    model.build_model(tf.AUTO_REUSE)
    
    ### create saver for restoring and saving variables
    saver = tf.train.Saver()
    
    with tf.Session() as sess:

        ### initialize global variable
        sess.run(tf.global_variables_initializer())
        
        ### restore checkpoint
        ckpt_path = _restore_checkpoint(saver,sess)
        
        ### load pre-trained vgg weights
        model.load_vgg_weight('vgg16_weights.npz', sess)
        
        ### lists of losses, used for tracking
        kl_loss = []
        pct_loss = []
        total_loss = []
        iteration = []
        
        ### count how many training iteration (different from epoch)
        iteration_count = 0
        
        for i in range(epoch):
            
            ### initialize iterator
            sess.run(iterator_init)
            
            try:
                while True:
                    sys.stdout.write('\r' + 'Iteration: ' + str(iteration_count))
                    sys.stdout.flush()
                    
                    ### every 100 iteration, print losses 
                    if iteration_count % 100 == 0:
                        pct, kl, loss, tempmean, tempstd = sess.run([model.pct_loss, model.kl_loss, model.loss, model.mean, model.std])
                        pct = pct * beta
                        print("\nperceptual loss: {}, kl loss: {}, total loss: {}".format(pct,kl,loss))
                        print(tempmean[0,0:5])
                        print(tempstd[0,0:5])
                        
                        iteration.append(iteration_count)
                        kl_loss.append(kl)
                        pct_loss.append(pct)
                        total_loss.append(loss)
                    
                    ### every 500 iteration, save images
                    if iteration_count % 500 == 0:
                        
                        ### get images from the dfc_vae_model
                        original_img, gen_img, ran_img = sess.run([model.img_input,model.gen_img,model.ran_img])
                        
                        ### denomalize
                        original_img = original_img*IMG_STDDEV + IMG_MEAN
                        gen_img = gen_img*IMG_STDDEV + IMG_MEAN
                        ran_img = ran_img*IMG_STDDEV + IMG_MEAN
                        
                        ### clip values to be in RGB range and transform to 0..1 float
                        original_img = np.clip(original_img,0.,255.).astype('float32')/255.
                        gen_img = np.clip(gen_img,0.,255.).astype('float32')/255.
                        ran_img = np.clip(ran_img,0.,255.).astype('float32')/255.
                        
                        ### save images in 5x5 grid
                        util.save_grid_img(original_img, os.path.join(os.getcwd(), 'outputs', str(i) + '_' + str(iteration_count) + '_orginal' + '.png'), img_height, img_width, 5, 5)
                        util.save_grid_img(gen_img, os.path.join(os.getcwd(), 'outputs', str(i) + '_' + str(iteration_count) + '_generated' + '.png'), img_height, img_width, 5, 5)
                        util.save_grid_img(ran_img, os.path.join(os.getcwd(), 'outputs', str(i) + '_' + str(iteration_count) + '_random' + '.png'), img_height, img_width, 5, 5)
                        
                        ### plot losses
                        plt.figure()
                        plt.plot(iteration, kl_loss)
                        plt.plot(iteration, pct_loss)
                        plt.plot(iteration, total_loss)
                        plt.legend(['kl loss', 'perceptual loss', 'total loss'], bbox_to_anchor=(1.05, 1), loc=2)
                        plt.title('Loss per iteration')
                        plt.show()
                    
                    ### run optimizer
                    sess.run(model.optimizer)
                    iteration_count += 1

            except tf.errors.OutOfRangeError:
                pass
            
            ### save session for each epoch
            ### recommend to change to save encoder, decoder, VGG's variables separately.
            print("\nepoch: {}, loss: {}".format(i, loss))
            saver.save(sess, os.path.join(ckpt_path,"Face_Vae"), global_step = iteration_count + model.gstep)
            print("checkpoint saved")
            

def test_gen_img(model, sess, i):
    
    real_img, gen_img, ran_img = sess.run([model.img_input, model.gen_img, model.ran_img])
    
    ran_img = ran_img*IMG_STDDEV + IMG_MEAN
    real_img = real_img*IMG_STDDEV + IMG_MEAN
    gen_img = gen_img*IMG_STDDEV + IMG_MEAN
    
    ran_img = ran_img/255.
    real_img = real_img/255.
    gen_img = gen_img/255.
    
    util.save_grid_img(ran_img, os.path.join(os.getcwd(), 'outputs', 'test' , 'random-' + str(i) + '.png'), 64,64,8,8)
    util.save_grid_img(real_img, os.path.join(os.getcwd(), 'outputs', 'test' , 'real-' + str(i) + '.png'), 64,64,8,8)
    util.save_grid_img(gen_img, os.path.join(os.getcwd(), 'outputs', 'test' , 'gen-' + str(i) + '.png'), 64,64,8,8)
    
def test_interpolation(model, sess, i):
    
    z1 = sess.run(model.z)
    z2 = sess.run(model.z)
    
    print(z1.shape)
    print(z2.shape)
    
    print(z1[0,:5])
    print(z2[0,:5])
    
    z = tf.Variable(np.zeros(z1.shape).astype(np.float32))
    gen_img = model.decoder(z, tf.AUTO_REUSE)
    
    interpolated_img_list = []
    
    for j in range(31):
        interpolated_z = z1 * (30-j)/30. + z2 * j/30. 
        sess.run(z.assign(interpolated_z))
        interpolated_img = sess.run(gen_img)
        interpolated_img = interpolated_img*IMG_STDDEV + IMG_MEAN
        interpolated_img = interpolated_img/255.
        interpolated_img = util.build_grid_img(interpolated_img, interpolated_img.shape[1], interpolated_img.shape[2],8,8)
        interpolated_img_list.append(interpolated_img)
        
    
    for j in range(31):
        imageio.mimsave(os.path.join(os.getcwd(), 'outputs', 'test_interpolated' , 'interpolate' + str(i) + '.gif'), interpolated_img_list)
    
    return interpolated_img_list
    
    
def test():

    ### setup hyper-parameter
    num_test_set = 2
    batch_size = 64
    vgg_layers = VGG_LAYERS
    img_height = IMG_HEIGHT
    img_width = IMG_WIDTH
    training_data_tfrecord = TRAINING_DATA
    
    ### get training data 
    iterator, _ = _get_data(training_data_tfrecord, batch_size)

    ### create iterator's initializer for training data 
    iterator_init = iterator.initializer
    data = iterator.get_next()
    
    ### define input data
    img_input = (tf.reshape(data, shape=[-1, img_height, img_width, 3]) - IMG_MEAN)/IMG_STDDEV
    
    ### build model graph
    model = dfc.dfc_vae_model([img_height,img_width], img_input)
    model.build_model(tf.AUTO_REUSE)
    
    ### create saver for restoring and saving variables
    saver = tf.train.Saver()
    
    with tf.Session() as sess:

        ### initialize global variable
        sess.run(tf.global_variables_initializer())
        
        ### restore checkpoint
        ckpt_path = _restore_checkpoint(saver,sess)

        sess.run(iterator_init)
        
        for i in range(num_test_set):
            
            test_gen_img(model, sess, i)
            x = test_interpolation(model, sess, i)
        

if __name__ == '__main__':
    tf.reset_default_graph()
    train_dfc_vae()
    test()

