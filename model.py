from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class WGAN_GP(object):
  #model_name = "WGAN_GP"     # name for checkpoint
  def __init__(self, sess, input_height=96, input_width=96, crop=False,
         batch_size=64, sample_num = 64, output_height=96, output_width=96,
         z_dim=100, gf_dim=64, df_dim=32,y_dim=None,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None,lambd=10):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.z_dim = z_dim
    self.y_dim = y_dim
	
    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim
    self.lambd = lambd
    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name
	
    self.input_fname_pattern = input_fname_pattern
	
    self.checkpoint_dir = checkpoint_dir

    self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
	
    imreadImg = imread(self.data[0]);
	
    if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
      self.c_dim = imread(self.data[0]).shape[-1]
    else:
      self.c_dim = 1

    self.grayscale = (self.c_dim == 1)

    self.build_model()

  def build_model(self):
    self.y = None
    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs

    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
    
    self.z_sum = histogram_summary("z", self.z)
    """ Loss Function """
    self.G                  = self.generator(self.z, self.y)
    # output of D for real images
    #self.D, self.D_logits   = self.discriminator(inputs, self.y, reuse=False)
    self.D_logits, _     = self.discriminator(inputs, self.y, reuse=False)
    # output of D for fake images
    self.sampler            = self.sampler(self.z, self.y)
	
    #self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
    self.D_logits_, self.D__   = self.discriminator(self.G, self.y, reuse=True)
    #self.sampler            = self.sampler(self.z, self.y)
	
	# final summary operations
    self.d_sum  = histogram_summary("d", self.D_logits)
    self.d__sum = histogram_summary("d_", self.D_logits_)
    self.G_sum  = image_summary("G", self.G)
	
    # get loss for discriminator
    self.d_loss_real = -tf.reduce_mean(self.D_logits)
    self.d_loss_fake = tf.reduce_mean(self.D_logits_)
    self.d_loss      = self.d_loss_real + self.d_loss_fake
    #get loss for generator
    self.g_loss      = -tf.reduce_mean(self.D_logits_)
	
    #self.d_sum = histogram_summary("d", self.D)
    #self.d__sum = histogram_summary("d_", self.D_)
    #self.G_sum = image_summary("G", self.G)
    """ Gradient Penalty """
    self.alpha = tf.random_uniform(shape=self.inputs.get_shape(), minval=0.1,maxval=1.)
    self.differences = self.G - self.inputs
    self.interpolates = self.inputs + (self.alpha * self.differences)
    self.D_inter,_=self.discriminator(self.interpolates,reuse=True)
    self.gradients = tf.gradients(self.D_inter, [self.interpolates])[0]
    self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), axis=[1]))#
    #self.gradient_penalty = tf.reduce_mean(tf.square(relu((self.slopes - 1.)))
    self.gradient_penalty = tf.reduce_mean(tf.square(tf.maximum(0.0,(self.slopes - 1.))))
    self.d_loss += self.lambd * self.gradient_penalty

    """" Testing """
    # for test
    #self.fake_images = self.sampler(self.z)
    """ Summary """
    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
    self.g_loss_sum      = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum      = scalar_summary("d_loss", self.d_loss)
    
    """ Training """
    # divide trainable variables into a group for D and a group for G
    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]
    self.saver = tf.train.Saver()
    ##################################################################################
    # train
    ##################################################################################
  def train(self, config):

    d_optim = tf.train.AdamOptimizer(config.learning_rate_d, beta1=config.beta1, beta2=0.9) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate_g, beta1=config.beta1, beta2=0.9) \
	          .minimize(self.g_loss, var_list=self.g_vars)

    tf.global_variables_initializer().run()

    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
	#self.g_sum = merge_summary([self.z_sum,self.d__sum,self.G_sum, self.d_loss_fake_sum,self.G_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    #self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum,self.d_loss_sum])
    self.writer = SummaryWriter("./logs", self.sess.graph)

    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
  
    sample_files = self.data[0:self.sample_num]
    sample = [
	  get_image(sample_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale) for sample_file in sample_files]
    if (self.grayscale):
      sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
    else:
      sample_inputs = np.array(sample).astype(np.float32)
  
    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):   
      self.data = glob(os.path.join("./data", config.dataset, self.input_fname_pattern))
      batch_idxs = min(len(self.data), config.train_size) // config.batch_size

      for idx in xrange(0, batch_idxs):
        batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
        batch = [get_image(batch_file,
                           input_height=self.input_height,
                           input_width=self.input_width,
						   resize_height=self.output_height,
                           resize_width=self.output_width,
                           crop=self.crop,
                           grayscale=self.grayscale) for batch_file in batch_files]
        if self.grayscale:
          batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
        else:
          batch_images = np.array(batch).astype(np.float32)

        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

        # Update D network
        _, summary_str,d_loss = self.sess.run([d_optim, self.d_sum, self.d_loss],feed_dict={ self.inputs: batch_images, self.z: batch_z })
        self.writer.add_summary(summary_str, counter)

        # Update G network
        _, summary_str,g_loss = self.sess.run([g_optim, self.g_sum,self.g_loss],feed_dict={ self.z: batch_z })
        self.writer.add_summary(summary_str, counter)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, summary_str,g_loss = self.sess.run([g_optim, self.g_sum, self.g_loss],feed_dict={ self.z: batch_z })
        self.writer.add_summary(summary_str, counter)
          
        #errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
        #errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
        #errG      = self.g_loss.eval({self.z: batch_z})
		
        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, batch_idxs,time.time() - start_time, d_loss, g_loss))

        if np.mod(counter, 100) == 1:
            samples, d_loss, g_loss = self.sess.run(
			[self.sampler, self.d_loss, self.g_loss],
			feed_dict={
			     self.z: sample_z,
				 self.inputs: sample_inputs,
				 },
			)
            save_images(samples, image_manifold_size(samples.shape[0]),
			     './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
		#tick()

        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)
  ##################################################################################
  # Discriminator
  ##################################################################################
  
  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()
      ch = self.df_dim*8
      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(conv2d(h0, self.df_dim*2, name='d_h1_conv'))
      h2 = lrelu(conv2d(h1, self.df_dim*4, name='d_h2_conv'))
      h3 = lrelu(conv2d(h2, self.df_dim*8, name='d_h3_conv'))
      h3 = self.attention(h3,ch)
      h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')
      return  h4 ,tf.nn.sigmoid(h4)
		
  ##################################################################################
  # Generator
  ##################################################################################
  
  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      s_h, s_w     = self.output_height, self.output_width
      s_h2, s_w2   = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4   = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8   = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
      ch = self.gf_dim
      # project `z` and reshape
      self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)
	  
      self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(self.h0))

      self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
      h1 = tf.nn.relu(self.g_bn1(self.h1))

      h2, self.h2_w, self.h2_b      = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
      h2 = tf.nn.relu(self.g_bn2(h2))

      h3, self.h3_w, self.h3_b      = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
      h3 = tf.nn.relu(self.g_bn3(h3))
      h3 = self.attention(h3, ch)
      h4, self.h4_w, self.h4_b      = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)
      return tf.nn.tanh(h4)

  def sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()
	  
      s_h, s_w     = self.output_height, self.output_width
      s_h2, s_w2   = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4   = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8   = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
      ch = self.gf_dim
      # project `z` and reshape
      h0 = tf.reshape(linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),[-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(h0, train=False))

      h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
      h1 = tf.nn.relu(self.g_bn1(h1, train=False))

      h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
      h2 = tf.nn.relu(self.g_bn2(h2, train=False))

      h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
      h3 = tf.nn.relu(self.g_bn3(h3, train=False))
	  
      h3 = self.attention(h3, ch)
	  
      h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

      return tf.nn.tanh(h4)
  def attention(self, x, ch):
        f = conv(x, ch // 8, kernel=1, stride=1, scope='f_conv')
        g = conv(x, ch // 8, kernel=1, stride=1, scope='g_conv')
        h = conv(x, ch, kernel=1, stride=1, scope='h_conv')

        s = tf.matmul(g, f, transpose_b=True)
        attention_shape = s.shape
        s = tf.reshape(s, shape=[attention_shape[0], -1, attention_shape[-1]])  # [bs, N, C]

        beta = tf.nn.softmax(s, axis=1)  # attention map
        beta = tf.reshape(beta, shape=attention_shape)
        o = tf.matmul(beta, h)

        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        x = gamma * o + x

        return x
  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(self.dataset_name, self.batch_size,self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name     = "WGAN_GP.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,os.path.join(checkpoint_dir, model_name),global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
