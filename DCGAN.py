import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Input, LeakyReLU, Conv2D, Dropout, Conv2DTranspose, MaxPooling2D
from keras.layers import Flatten, BatchNormalization, Dense, Activation
from keras.optimizers import Adam
import configparser


def build_generator():
    model = Sequential()

    model.add(Conv2D(128, 5, padding='same', use_bias=False, name='g_conv_1', input_shape=(None, None, 3)))
    model.add(BatchNormalization(momentum=0.8, name='g_bn_2'))
    model.add(Activation('relu', name='g_relu_2'))

    model.add(Conv2DTranspose(128, 3, padding='same', strides=2, name='g_upsample_1', activation='relu'))

    model.add(Conv2D(64, 3, padding='same', use_bias=False, name='g_conv_2'))
    model.add(BatchNormalization(momentum=0.8, name='g_bn_3'))
    model.add(Activation('relu', name='g_relu_3'))

    model.add(Conv2D(32, 3, padding='same', use_bias=False, name='g_conv_3'))
    model.add(BatchNormalization(momentum=0.8, name='g_bn_4'))
    model.add(Activation('relu', name='g_relu_4'))

    model.add(Conv2DTranspose(32, 3, padding='same', strides=2, name='g_upsample_2', activation='relu'))
    model.add(Conv2DTranspose(32, 3, padding='same', strides=1, name='d_denoise', activation='relu'))

    model.add(Conv2D(16, 3, padding='same', use_bias=False, name='g_conv_4'))
    model.add(BatchNormalization(momentum=0.8, name='g_bn_5'))
    model.add(Activation('relu', name='g_relu_5'))

    model.add(Conv2D(3, 3, padding='same', name='g_conv_7'))
    model.add(Activation('tanh', name='g_score'))

    noise = Input(shape=(None, None, 3), name='g_input')
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


if __name__ == '__main__':

    from keras.engine.network import Network

    image_shape = (96, 64, 3)
    latent_dim = 64
    stage = 2

    optimizer = Adam(lr=0.0002, beta_1=0.5)
    dropout_rate = 0.2

    discriminator_input, discriminator_output = build_discriminator(image_shape, dropout_rate)

    discriminator = Model(
        discriminator_input,
        discriminator_output,
        name='discriminator'
    )

    discriminator.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    assert (len(discriminator._collected_trainable_weights) > 0)

    frozen_discriminator = Network(
        discriminator_input,
        discriminator_output,
        name='frozen_discriminator'
    )
    frozen_discriminator.trainable = False

    generator_input, generator_output = build_generator()
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

    adversarial_model.compile(optimizer, loss='binary_crossentropy')

    assert (len(adversarial_model._collected_trainable_weights) == len(generator.trainable_weights))
    batch_size = 64

    config = configparser.ConfigParser()
    config.read('umikry.ini')
    if 'DATA' in config:
        data_dir = config['DATA']['Location']
    else:
        data_dir = input('Choose a directory to store the data:')
        config['DATA'] = {'Location': data_dir}

        with open('umikry.ini', 'w') as configfile:
            config.write(configfile)

    train_dir = os.path.join(data_dir, '/WIDERFace_Haar/positives/')

    def SampleGenerator(batch_size=32, train_folder=train_dir):
        possible_images = os.listdir(train_folder)

        while True:
            images = np.zeros((batch_size, 96, 64, 3))
            random_images = random.sample(possible_images, batch_size)
            for i in range(batch_size):
                image = cv2.imread(os.path.join(train_folder, random_images[i]))
                b, g, r = cv2.split(image)
                image = cv2.merge([r, g, b])
                images[i] = cv2.resize(image, (64, 96)) / 255.0

            yield np.array(images)

    sampleGenerator = SampleGenerator(batch_size)

    epochs = 160000
    plot_rows = 5
    plot_cols = 5

    sampleTestGenerator = SampleGenerator(plot_rows * plot_cols)
    output_location = os.path.join('gan_output')

    for epoch in range(epochs):
        images = next(sampleGenerator)
        pseudo_noise = []
        for image in images:
            pseudo_noise.append(cv2.resize(image, (16, 24)))

        noise = np.array(pseudo_noise)
        generated_images = generator.predict(noise)
        discriminator_loss_valid = discriminator.train_on_batch(images, np.ones((len(images), 1)))
        discriminator_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))

        discriminator_loss = np.add(discriminator_loss_fake, discriminator_loss_valid) * 0.5

        generator_loss = adversarial_model.train_on_batch(noise, np.ones((batch_size, 1)))

        if epoch % 300 == 0:
            images = next(sampleTestGenerator)
            pseudo_noise = []
            for image in images:
                pseudo_noise.append(cv2.resize(image, (16, 24)))

            noise = np.array(pseudo_noise)
            generated_images = generator.predict(noise)
            generated_images = generated_images * 0.5 + 0.5
            fig, axis = plt.subplots(plot_rows, plot_cols)

            for i in range(plot_rows):
                for j in range(plot_cols):
                    axis[i, j].imshow(generated_images[i * plot_cols + j])
                    axis[i, j].axis('off')

            fig.savefig(os.path.join(output_location, 'images', 'epoch_{}.png'.format(epoch)))
            plt.show()
            generator_save_path = os.path.join(output_location, 'generator.h5')
            discriminator_save_path = os.path.join(output_location, 'discriminator.h5')
            if os.path.isfile(generator_save_path):
                os.remove(generator_save_path)

            if os.path.isfile(discriminator_save_path):
                os.remove(discriminator_save_path)

            discriminator.save(discriminator_save_path)
            generator.save(generator_save_path)
            print('{} [D loss: {:.6f}, acc.: {:.2f}%] [G loss: {:.6f}]'.format(
                epoch, discriminator_loss[0], 100 * discriminator_loss[1], generator_loss))
