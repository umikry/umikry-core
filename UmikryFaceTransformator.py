import cv2
import numpy as np
import math
import os
from UmikryDCGAN import build_generator
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D


def next_multiply_of_n(x, n):
    return math.ceil(x / n) * n


def pad_to_next_multiply_of_n(image, n):
    diff_y = next_multiply_of_n(image.shape[0], n) - image.shape[0]
    diff_x = next_multiply_of_n(image.shape[1], n) - image.shape[1]

    border_top = int(diff_y / 2)
    border_bottom = diff_y - border_top
    border_left = int(diff_x / 2)
    border_right = diff_x - border_left

    padded_image = cv2.copyMakeBorder(image, border_top, border_bottom, border_left, border_right,
                                      cv2.BORDER_REFLECT)
    return padded_image, (border_top, border_bottom, border_left, border_right)


class UmikryFaceTransformator():
    def __init__(self, method='AUTOENCODER'):
        if method == 'AUTOENCODER':
            self.method = method
            self.model = self.build_autoencoder()
            self.model.load_weights(os.path.join('models', 'autoencoder_weights.h5'), by_name=True)
        elif method == 'GAN':
            self.method = method
            generator_input, generator_output = build_generator()
            self.model = Model(generator_input, generator_output)
            self.model.load_weights(os.path.join('models', 'generator_weights.h5'))
        elif method == 'BLUR':
            self.method = method
        else:
            raise ValueError('{} is an invalid method use \'AUTOENCODER\', \'GAN\' or \'BLUR\' instead'.format(self.method))

    def build_autoencoder(self):
        input_image = Input(shape=(None, None, 3), name='e_input')
        x = Conv2D(32, 3, padding='same', name='e_conv1', activation='relu')(input_image)
        x = MaxPooling2D(padding='same', name='e_pool1')(x)
        x = Conv2D(64, 3, padding='same', name='e_conv2', activation='relu')(x)
        x = MaxPooling2D(padding='same', name='e_pool2')(x)
        x = Conv2D(128, 3, padding='same', name='e_conv3', activation='relu')(x)
        x = MaxPooling2D(padding='same', name='e_pool3')(x)
        encoder = Conv2D(128, 3, padding='same', name='encoded')(x)

        x = Conv2D(128, 3, padding='same', name='d_conv1', activation='relu')(encoder)
        x = Conv2DTranspose(128, 3, padding='same', strides=2, name='d_up1', activation='relu')(x)
        x = Conv2D(64, 3, padding='same', name='d_conv2', activation='relu')(x)
        x = Conv2DTranspose(64, 3, padding='same', strides=2, name='d_up2', activation='relu')(x)
        x = Conv2D(32, 3, padding='same', name='d_conv3', activation='relu')(x)
        x = Conv2DTranspose(32, 3, padding='same', strides=2, name='d_up3', activation='relu')(x)
        x = Conv2DTranspose(32, 3, padding='same', strides=1, name='d_denoise', activation='relu')(x)
        x = Conv2D(3, 5, padding='same', name='d_pre_decoded', activation='sigmoid')(x)
        decoder = Conv2D(3, 1, padding='same', name='decoded', activation='sigmoid')(x)

        return Model(inputs=input_image, outputs=decoder)

    def transform(self, image, faces):
        if self.method == 'BLUR':
            for _, (x, y, w, h) in faces.items():
                if h < image.shape[0] and w < image.shape[1]:
                    image[y:h, x:w] = cv2.blur(image[y:h, x:w], (25, 25))

            return image
        elif self.method == 'AUTOENCODER':
            for _, (x, y, w, h) in faces.items():
                if h < image.shape[0] and w < image.shape[1]:
                    blurry = cv2.blur(image[y:h, x:w], (25, 25)) / 255.0
                    blurry, border = pad_to_next_multiply_of_n(blurry, 8)

                    prediction = self.model.predict(np.array([blurry]))[0]
                    prediction = prediction[border[0]:(prediction.shape[0] - border[1]),
                                            border[2]:(prediction.shape[1] - border[3]), :]

                    image[y:h, x:w] = (prediction * 255).astype(np.uint8)

            return image
        elif self.method == 'GAN':
            for _, (x, y, w, h) in faces.items():
                if h < image.shape[0] and w < image.shape[1]:
                    section, border = pad_to_next_multiply_of_n(image[y:h, x:w], 4)
                    section = cv2.resize(section, (section.shape[1] // 4, section.shape[0] // 4)) / 255.0

                    prediction = self.model.predict(np.array([section]))[0] * 0.5 + 0.5

                    prediction = (prediction[border[0]:(prediction.shape[0] - border[1]),
                                             border[2]:(prediction.shape[1] - border[3]), :] * 255)

                    prediction = prediction.astype(np.uint8)
                    face = self.__smoth_replace(image[y:h, x:w], prediction)
                    image[y:h, x:w] = face

            return image

    def __smoth_replace(self, image, face):
        mean_values = np.mean(np.mean(image, axis=0), axis=0)
        mean_shift = mean_values - np.mean(np.mean(face, axis=0), axis=0)

        for i in range(face.shape[2]):
            face[:, :, i] = face[:, :, i] + mean_shift[i]
            face[:, :, i] = np.where(face[:, :, i] > 0, face[:, :, i], 0)
            face[:, :, i] = np.where(face[:, :, i] < 255, face[:, :, i], 255)

        steps = [(0, 0.2), (2, 0.4), (4, 0.5), (8, 0.6), (10, 1.0)]
        smoth_face = np.ones_like(face)
        for step, intensity in steps:
            if step == 0:
                background = np.uint8(image * (1 - intensity))
                foreground = np.uint8(face * intensity)
                smoth_face = background + foreground
            else:
                background = np.uint8(image[step:-step, step:-step] * (1 - intensity))
                foreground = np.uint8(face[step:-step, step:-step] * intensity)
                smoth_face[step:-step, step:-step] = background + foreground

        return smoth_face
