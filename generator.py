from keras.models import Model, Sequential
from keras.layers import Input, LeakyReLU, Conv2D, Dropout, Conv2DTranspose, Reshape
from keras.layers import Flatten, BatchNormalization, Dense, Activation
from keras.engine.network import Network
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np
import configparser


def build_generator(latent_dim):
  model = Sequential()

  model.add(Dense(128 * 8 * 8, input_dim=latent_dim, use_bias=False, name='g_dense'))
  model.add(Reshape((8, 8, 128), name='g_reshape'))
  model.add(BatchNormalization(momentum=0.8, name='g_bn_1'))
  model.add(Activation('relu', name='g_relu_1'))
  
  model.add(UpSampling2D(name='g_upsample_1'))
  model.add(Conv2D(256, 5, padding='same', use_bias=False, name='g_conv_1'))
  model.add(BatchNormalization(momentum=0.8, name='g_bn_2'))
  model.add(Activation('relu', name='g_relu_2'))

  model.add(UpSampling2D(name='g_upsample_2'))
  model.add(Conv2D(128, 5, padding='same', use_bias=False, name='g_conv_2'))
  model.add(BatchNormalization(momentum=0.8, name='g_bn_3'))
  model.add(Activation('relu', name='g_relu_3'))

  model.add(UpSampling2D(name='g_upsample_3'))
  model.add(Conv2D(64, 5, padding='same', use_bias=False, name='g_conv_3'))
  model.add(BatchNormalization(momentum=0.8, name='g_bn_4'))
  model.add(Activation('relu', name='g_relu_4'))

  model.add(UpSampling2D(name='g_upsample_4'))
  model.add(Conv2D(32, 3, padding='same', use_bias=False, name='g_conv_4'))
  model.add(BatchNormalization(momentum=0.8, name='g_bn_5'))
  model.add(Activation('relu', name='g_relu_5'))
  
  model.add(UpSampling2D(name='g_upsample_5'))
  model.add(Conv2D(16, 3, padding='same', use_bias=False, name='g_conv_5'))
  model.add(BatchNormalization(momentum=0.8, name='g_bn_6'))
  model.add(Activation('relu', name='g_relu_6'))

  model.add(Conv2D(3, 3, padding='same', name='g_conv_7'))
  model.add(Activation('tanh', name='g_score'))
  
  noise = Input(shape=(latent_dim,), name='g_input')
  image = model(noise)

  return noise, image


def build_discriminator(image_shape, dropout_rate):
  model = Sequential()
  
  model.add(Conv2D(16, 3, input_shape=image_shape, padding='same', name='d_conv_1'))
  model.add(LeakyReLU(0.1, name='d_relu_1'))
  model.add(MaxPooling2D(name='d_pool1'))
  model.add(Dropout(dropout_rate))
  
  model.add(Conv2D(32, 3, padding='same', use_bias=False, name='d_conv_2'))
  model.add(BatchNormalization(momentum=0.8, name='d_bn_1'))
  model.add(LeakyReLU(0.1, name='d_relu_2'))
  model.add(MaxPooling2D(name='d_pool2'))
  model.add(Dropout(dropout_rate))
  
  model.add(Conv2D(64, 3, padding='same', use_bias=False, name='d_conv_3'))
  model.add(BatchNormalization(momentum=0.8, name='d_bn_2'))
  model.add(LeakyReLU(0.1, name='d_relu_3'))
  model.add(MaxPooling2D(name='d_pool3'))
  model.add(Dropout(dropout_rate))
  
  model.add(Conv2D(128, 3, padding='same', use_bias=False, name='d_conv_4'))
  model.add(BatchNormalization(momentum=0.8, name='d_bn_3'))
  model.add(LeakyReLU(0.1, name='d_relu_4'))
  model.add(MaxPooling2D(name='d_pool4'))
  model.add(Dropout(dropout_rate))
  
  model.add(Conv2D(256, 3, padding='same', use_bias=False, name='d_conv_5'))
  model.add(BatchNormalization(momentum=0.8, name='d_bn_4'))
  model.add(LeakyReLU(0.1, name='d_relu_5'))
  model.add(MaxPooling2D(name='d_pool5'))
  model.add(Dropout(dropout_rate))
  
  model.add(Flatten(name='d_flatten'))
  model.add(Dense(1, activation='sigmoid', name='d_score'))

  image = Input(shape=image_shape, name='d_input')
  validity = model(image)

  return image, validity


config = configparser.ConfigParser()
config.read('umikry.ini')
    
if 'DATA' in config:
  data_dir = config['DATA']['Location']
else:
  data_dir = input('Where is your train data located:')
  config['DATA'] = {'Location': data_dir}

  with open('umikry.ini', 'w') as configfile:
    config.write(configfile)
    
data_location = os.path.join(data_dir, 'GAN')
output_location = 'gan_output'
if not os.path.isdir(output_location):
  os.makedirs(os.path.join(output_location, 'images'))

image_shape = (256, 256, 3)
latent_dim = 128
stage = 2

if stage == 1:
  optimizer = Adam(lr=0.0002, beta_1=0.5)
  dropout_rate = 0.3
else:
  optimizer = Adam(lr=0.00005, beta_1=0.3)
  dropout_rate = 0.1

discriminator_input, discriminator_output = build_discriminator(image_shape, dropout_rate)
    
discriminator = Model(
  discriminator_input, 
  discriminator_output,
  name='discriminator'
)

discriminator.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])

assert(len(discriminator._collected_trainable_weights) > 0)

frozen_discriminator = Network(
  discriminator_input,
  discriminator_output,
  name='frozen_discriminator'
)
frozen_discriminator.trainable = False

generator_input, generator_output = build_generator(latent_dim)
generator = Model(
  generator_input, 
  generator_output,
  name='generator'
)

# restore generator stage 1 weights
if stage == 2:
  generator.load_weights(os.path.join(output_location, 'generator_stage1.h5'))
  discriminator.load_weights(os.path.join(output_location, 'discriminator_stage1.h5'))
    
adversarial_model = Model(
  generator_input,
  frozen_discriminator(generator_output),
  name='adversarial_model'
)

adversarial_model.compile(optimizer, loss='binary_crossentropy')

assert(len(adversarial_model._collected_trainable_weights) == len(generator.trainable_weights))    
batch_size=48

image_data_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
image_generator = image_data_generator.flow_from_directory(
  data_location,
  batch_size=(batch_size),
  target_size=(256, 256),
  class_mode=None
)

epochs = 40000
plot_rows = 5
plot_cols = 5

for epoch in range(epochs):
  images = next(image_generator)
  noise = np.random.normal(0, 1, (batch_size, latent_dim))
  generated_images = generator.predict(noise)

  discriminator_loss_valid = discriminator.train_on_batch(images, np.ones((len(images), 1)))
  discriminator_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))
  
  discriminator_loss = np.add(discriminator_loss_fake, discriminator_loss_valid) * 0.5
  
  generator_loss = adversarial_model.train_on_batch(noise, np.ones((batch_size, 1)))

  if epoch % 100 == 0:
    noise = np.random.normal(0, 1, (plot_rows * plot_cols, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = generated_images * 0.5 + 0.5
    fig, axis = plt.subplots(plot_rows, plot_cols)

    for i in range(plot_rows):
      for j in range(plot_cols):
        axis[i, j].imshow(generated_images[i * plot_cols + j])
        axis[i, j].axis('off')
      
    fig.savefig(os.path.join(output_location, 'images', 'epoch_{}.png'.format(epoch)))
    plt.show()
    generator_save_path = os.path.join(output_location, 'generator_stage{}.h5'.format(stage))
    discriminator_save_path = os.path.join(output_location, 'discriminator_stage{}.h5'.format(stage))
    if os.path.isfile(generator_save_path):
      os.remove(generator_save_path)

    if os.path.isfile(discriminator_save_path):
      os.remove(discriminator_save_path)

    discriminator.save(discriminator_save_path)
    generator.save(generator_save_path)             
    print('{} [D loss: {:.6f}, acc.: {:.2f}%] [G loss: {:.6f}]'.format(
      epoch, discriminator_loss[0], 100 * discriminator_loss[1], generator_loss))
