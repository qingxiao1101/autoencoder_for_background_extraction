import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from models import *
from data import Data
from ssim import tf_ssim
from skimage import io

class ConvolutionalAutoencoder(object):
    """
    """
    def __init__(self):
        """
        build the graph
        """
        # place holder of input data
        x = tf.placeholder(tf.float32, shape=[None, 300, 480, 1])  # [#batch, img_height, img_width, #channels]

        # encode
        conv1 = Convolution2D([5, 5, 1, 32], activation=tf.nn.relu, scope='conv_1')(x)
        pool1 = MaxPooling(kernel_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', scope='pool_1')(conv1)
        conv2 = Convolution2D([5, 5, 32, 32], activation=tf.nn.relu, scope='conv_2')(pool1)
        pool2 = MaxPooling(kernel_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', scope='pool_2')(conv2)
        unfold = Unfold(scope='unfold')(pool2)
        encoded = FullyConnected(20, activation=tf.nn.relu, scope='encode')(unfold)
        # decode
        decoded = FullyConnected(75*120*32, activation=tf.nn.relu, scope='decode')(encoded)
        fold = Fold([-1, 75, 120, 32], scope='fold')(decoded)
        unpool1 = UnPooling((2, 2), output_shape=tf.shape(conv2), scope='unpool_1')(fold)
        deconv1 = DeConvolution2D([5, 5, 32, 32], output_shape=tf.shape(pool1), activation=tf.nn.relu, scope='deconv_1')(unpool1)
        unpool2 = UnPooling((2, 2), output_shape=tf.shape(conv1), scope='unpool_2')(deconv1)
        reconstruction = DeConvolution2D([5, 5, 1, 32], output_shape=tf.shape(x), activation=tf.nn.sigmoid, scope='deconv_2')(unpool2)

        # loss function
        loss = 1 - tf_ssim(x, reconstruction) #l2_loss + 1000*ssim_loss

        # training
        training = tf.train.AdamOptimizer(1e-4).minimize(loss)

        self.x = x
        self.reconstruction = reconstruction
        self.loss = loss
        self.training = training


    def train(self, passes, image_path):
        """

        :param batch_size:
        :param passes:
        :param new_training:
        :return:
        """
        data = Data()
        images = data.get_image(file_path = image_path)

        with tf.Session() as sess:
           
            saver, global_step = Model.start_new_session(sess)

            for step in range(1+global_step, 1+passes+global_step):

                for image in images:

                    image = image[np.newaxis, :, :, np.newaxis]
                    #print(f'image shape is: {image.shape}')
                    self.training.run(feed_dict={self.x: image})
                    
                    #l2_loss = self.conv.train.run([self.l2_loss], feed_dict={self.x: image})

                #background = np.reshape(background, (120, 300))

                if step % 1 == 0:
                    loss = self.loss.eval(feed_dict={self.x: image})
                    print("pass {}, training loss {}".format(step, loss))
                    pass

                if step % 10 == 0:  # save weights
                    saver.save(sess, 'saver/cnn', global_step=step)
                    pass
                
    
    def reconstruct(self, image_path, background_saved_path, test_original):
        """

        """
        data = Data()
        print(f'image_path: {image_path}')
        images = data.get_image(file_path = image_path)
        print(f'images: {images.shape}')

        with tf.Session() as sess:
            saver, global_step = Model.continue_previous_session(sess, ckpt_file='saver/checkpoint')

            backgrounds = []
            for image in images:
                image = image[np.newaxis, :, :, np.newaxis]
                background = sess.run(self.reconstruction, feed_dict={self.x: image})
                background = background.reshape((background.shape[1], background.shape[2]))
                backgrounds.append(background)
            
            for index, background in enumerate(backgrounds):
                io.imsave(background_saved_path + '/' +str(index) + '.jpg', backgrounds[index])
                io.imsave(test_original + '/' +str(index) + '.jpg', images[index])

def train_test(train_flag = True):
    train_image_path = './train_image'
    background_path = './background'
    test_image_path = './test_image'
    test_image_gray = './test_gray'

    conv_autoencoder = ConvolutionalAutoencoder()
    if train_flag:
        conv_autoencoder.train(passes = 10, image_path = train_image_path)
    else:
        conv_autoencoder.reconstruct(image_path = test_image_path, background_saved_path = background_path, test_original = test_image_gray)

if __name__ == '__main__':
    train_test(False)
