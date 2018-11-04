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

  model.add(Dense(128 * 8 * 8, input_dim=latent_dim, use_bias=False))
  model.add(Reshape((8, 8, 128)))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Activation('relu'))

  model.add(Conv2DTranspose(128, 5, strides=2, padding='same', use_bias=False))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Activation('relu'))

  model.add(Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Activation('relu'))

  model.add(Conv2DTranspose(32, 5, strides=2, padding='same', use_bias=False))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Activation('relu'))

  model.add(Conv2DTranspose(16, 5, strides=2, padding='same', use_bias=False))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Activation('relu'))

  model.add(Conv2DTranspose(8, 5, strides=2, padding='same', use_bias=False))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Activation('relu'))

  model.add(Conv2DTranspose(3, 5, padding='same'))
  model.add(Activation('tanh'))

  noise = Input(shape=(latent_dim,))
  image = model(noise)

  return noise, image


def build_discriminator(image_shape):
    model = Sequential()

    model.add(Conv2D(8, 5, strides=2, input_shape=image_shape, padding='same'))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(16, 5, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, 5, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, 5, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, 5, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    image = Input(shape=image_shape)
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

discriminator_input, discriminator_output = build_discriminator(image_shape)
discriminator = Model(
  discriminator_input,
  discriminator_output,
  name='discriminator'
)

discriminator.compile(Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
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

adversarial_model = Model(
  generator_input,
  frozen_discriminator(generator_output),
  name='adversarial_model'
)

adversarial_model.compile(Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')

assert(len(adversarial_model._collected_trainable_weights) == len(generator.trainable_weights))
batch_size = 64

valid_labels = np.ones((batch_size // 2, 1))
fake_labels = np.zeros((batch_size // 2, 1))

image_data_generator = ImageDataGenerator(rescale=(1. / 255), horizontal_flip=True)
image_generator = image_data_generator.flow_from_directory(
    data_location,
    batch_size=(batch_size // 2),
    target_size=(256, 256),
    class_mode=None
)

epochs = 10000
plot_rows = 5
plot_cols = 5

for epoch in range(epochs):
  images = next(image_generator)
  noise = np.random.normal(0, 1, (batch_size, latent_dim))
  generated_images = generator.predict(noise[:(batch_size // 2)])

  discriminator_loss_valid = discriminator.train_on_batch(images, valid_labels)
  discriminator_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)

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

    if os.path.isfile(os.path.join(output_location, 'generator.h5')):
      os.remove(os.path.join(output_location, 'generator.h5'))

    if os.path.isfile(os.path.join(output_location, 'discriminator.h5')):
      os.remove(os.path.join(output_location, 'discriminator.h5'))

    discriminator.save(os.path.join(output_location, 'discriminator.h5'))
    generator.save(os.path.join(output_location, 'generator.h5'))
    print('{} [D loss: {:.6f}, acc.: {:.2f}%%] [G loss: {:.6f}'.format(
          epoch, discriminator_loss[0], 100 * discriminator_loss[1], generator_loss))
