from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Dropout, Conv2DTranspose, Reshape, Flatten, BatchNormalization, Dense, Activation    
import os
from keras.utils import Sequence
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.preprocessing.image import ImageDataGenerator
import random
import matplotlib.pyplot as plt
import numpy as np
import configparser

def build_generator(latent_dim):
    model = Sequential()

    model.add(Dense(128 * 8 * 8, input_dim=latent_dim, kernel_initializer=RandomNormal(stddev=0.02), use_bias=False))
    model.add(Reshape((8, 8, 128)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('elu'))
    
    model.add(Conv2DTranspose(128, 5, strides=2, padding='same', kernel_initializer=RandomNormal(stddev=0.02), use_bias=False))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('elu'))

    model.add(Conv2DTranspose(64, 5, strides=2, padding='same', kernel_initializer=RandomNormal(stddev=0.02), use_bias=False))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('elu'))

    model.add(Conv2DTranspose(32, 5, strides=2, padding='same', kernel_initializer=RandomNormal(stddev=0.02), use_bias=False))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('elu'))

    model.add(Conv2DTranspose(16, 5, strides=2, padding='same', kernel_initializer=RandomNormal(stddev=0.02), use_bias=False))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('elu'))
    
    model.add(Conv2DTranspose(8, 5, strides=2, padding='same', kernel_initializer=RandomNormal(stddev=0.02), use_bias=False))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('elu'))
        
    model.add(Conv2DTranspose(3, 5, strides=1, padding='same', kernel_initializer=RandomNormal(stddev=0.02), activation='tanh'))

    noise = Input(shape=(latent_dim,))
    image = model(noise)

    return Model(noise, image)

def build_discriminator(image_shape):
    model = Sequential()

    model.add(Conv2D(32, 5, strides=2, input_shape=image_shape, kernel_initializer=RandomNormal(stddev=0.02), padding="same"))
    model.add(Activation('elu'))

    model.add(Conv2D(32, 5, strides=2, padding="same", kernel_initializer=RandomNormal(stddev=0.02), use_bias=False))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('elu'))

    model.add(Conv2D(64, 5, strides=2, padding="same", kernel_initializer=RandomNormal(stddev=0.02), use_bias=False))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('elu'))
    
    model.add(Conv2D(64, 5, strides=2, padding="same", kernel_initializer=RandomNormal(stddev=0.02), use_bias=False))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('elu'))

    model.add(Conv2D(128, 5, strides=2, padding="same", kernel_initializer=RandomNormal(stddev=0.02), use_bias=False))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('elu'))
    
    model.add(Flatten())
    model.add(Dense(1, kernel_initializer=RandomNormal(stddev=0.02), activation='sigmoid'))

    image = Input(shape=image_shape)
    validity = model(image)

    return Model(image, validity)
  
if __name__ == '__main__':
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

  discriminator = build_discriminator(image_shape)
  discriminator.trainable = False

  generator = build_generator(latent_dim)

  random_input = Input(shape=(latent_dim,))
  image = generator(random_input)
  
  is_valid = discriminator(image)

  combined = Model(random_input, is_valid)
  combined.compile(loss='binary_crossentropy', optimizer='adam')

  discriminator.trainable = True
  discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  
  batch_size=64

  valid_labels = np.ones((batch_size // 2, 1))
  fake_labels = np.zeros((batch_size // 2, 1))

  image_data_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=15)
  image_generator = image_data_generator.flow_from_directory(
    data_location,
    batch_size=(batch_size // 2),
    target_size=(256, 256),
    class_mode=None
  )

  epochs = 10000
  plot_rows = 5
  plot_cols = 5
  fixed_noise = np.random.normal(0, 1, (plot_rows * plot_cols, latent_dim))

  for epoch in range(epochs):
    images = next(image_generator)
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    generated_images = generator.predict(noise[:(batch_size // 2)])

    mixed_images = np.concatenate([generated_images, images], axis=0)
    mixed_labels = np.concatenate([fake_labels, valid_labels], axis=0)

    permutation = np.random.permutation(len(mixed_images))
    mixed_images = mixed_images[permutation]
    mixed_labels = mixed_labels[permutation]

    discriminator_loss = discriminator.train_on_batch(mixed_images, mixed_labels)
    generator_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))

    if epoch % 100 == 0:
      generated_images = generator.predict(fixed_noise)
      generated_images = generated_images * 0.5 + 0.5
      fig, axis = plt.subplots(plot_rows, plot_cols)
   
      for i in range(plot_rows):
        for j in range(plot_cols):
          axis[i, j].imshow(generated_images[i * j + j])
          axis[i, j].axis('off')
            
      fig.savefig(os.path.join(output_location, 'images', 'epoch_{}.png'.format(epoch)))
      
      if os.path.isfile(os.path.join(output_location, 'generator.h5')):
        os.remove(os.path.join(output_location, 'generator.h5'))

      if os.path.isfile(os.path.join(output_location, 'discriminator.h5')):
        os.remove(os.path.join(output_location, 'discriminator.h5'))

      discriminator.save(os.path.join(output_location, 'discriminator.h5'))
      generator.save(os.path.join(output_location, 'generator.h5'))             
      print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, discriminator_loss[0], 100*discriminator_loss[1], generator_loss))
