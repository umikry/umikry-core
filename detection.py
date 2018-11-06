from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add, BatchNormalization, LeakyReLU, Conv2DTranspose
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.utils import Sequence
import numpy as np
import configparser
import os
import random
import sys
import math
import time
import cv2


def generateTrainTestSet(data_location, datasets, test_size=0.1, override=True):
  if not os.path.isdir(data_location):
    sys.exit(data_location + ' does not exist. Hint: Use data.py to load the datasets.')

  if override:
    if os.path.isfile(os.path.join(data_location, 'train.txt')):
      os.unlink(os.path.join(data_location, 'train.txt'))
    if os.path.isfile(os.path.join(data_location, 'test.txt')):
      os.unlink(os.path.join(data_location, 'test.txt'))

  for dataset in datasets:
    available_labels = os.listdir(os.path.join(data_location, dataset, 'faces_center_cropped_label'))

    for filename in available_labels:
      data = os.path.join(dataset, 'faces_center_cropped', filename)
      label = os.path.join(dataset, 'faces_center_cropped_label', filename)
      if random.random() > test_size:
        with open(os.path.join(data_location, 'train.txt'), 'a') as trainlist:
          trainlist.write(data + ',' + label + '\n')
      else:
        with open(os.path.join(data_location, 'test.txt'), 'a') as testlist:
          testlist.write(data + ',' + label + '\n')


def next_multiply_of_n(x, n):
  return math.ceil(x / n) * n


def pad_to(image, n, axis=0):
  diff = n - image.shape[axis]

  if axis == 0:
    border_top = int(diff / 2)
    border_bottom = diff - border_top
    return (cv2.copyMakeBorder(image, border_top, border_bottom, 0, 0,
                             cv2.BORDER_CONSTANT, value=0),
          (border_top, border_bottom, 0, 0))
  if axis == 1:
    border_left = int(diff / 2)
    border_right = diff - border_left
    return (cv2.copyMakeBorder(image, 0, 0, border_left, border_right,
                             cv2.BORDER_CONSTANT, value=0),
          (0, 0, border_left, border_right))

def pad_to_next_multiply_of_n(image, n):
  diff_y = next_multiply_of_n(image.shape[0], n) - image.shape[0]
  diff_x = next_multiply_of_n(image.shape[1], n) - image.shape[1]

  border_top = int(diff_y / 2)
  border_bottom = diff_y - border_top
  border_left = int(diff_x / 2)
  border_right = diff_x - border_left

  padded_image = cv2.copyMakeBorder(image, border_top, border_bottom, border_left, border_right, 
                                    cv2.BORDER_CONSTANT, value=0)
  return padded_image, (border_top, border_bottom, border_left, border_right)


def crop_random(image, truth, size=(512, 512)):
  if image.shape[0] < size[0]:
    image, _ = pad_to(image, size[0], axis=0)
    truth, _ = pad_to(truth, size[0], axis=0)
  if image.shape[1] < size[1]:
    image, _ = pad_to(image, size[1], axis=1)
    truth, _ = pad_to(truth, size[1], axis=1)

  if image.shape[0] > size[0]:
    crop_random_y = random.randint(0, image.shape[0] - size[0])
    image = image[crop_random_y:crop_random_y + size[0], :, :]
    truth = truth[crop_random_y:crop_random_y + size[0], :]
  if image.shape[1] > size[1]:
    crop_random_x = random.randint(0, image.shape[1] - size[1])
    image = image[:, crop_random_x:crop_random_x + size[1], :]
    truth = truth[:, crop_random_x:crop_random_x + size[1]]

  return image, truth.reshape(image.shape[0], image.shape[1], 1)

class ImageSequence(Sequence):
  def __init__(self, data_location, batch_size=32, is_training=False, patch_size=(512, 512), plain_images=False):
    if is_training:
      self.dataset = open(os.path.join(data_location, 'train.txt')).readlines()
    else:
      self.dataset = open(os.path.join(data_location, 'test.txt')).readlines()

    self.data_location = data_location
    self.batch_size = batch_size
    self.patch_size = patch_size
    self.plain_images = plain_images

  def __len__(self):
    return len(self.dataset) // self.batch_size

  def __getitem__(self, i):
    if self.plain_images:
      sample = self.dataset[i]
      image_file = sample.split(',')[0]
      truth_file = sample.split(',')[1][:-1]        
      image = np.float32(cv2.imread(os.path.join(self.data_location, image_file)) / 255.0)
      truth = cv2.imread(os.path.join(self.data_location, truth_file), cv2.IMREAD_GRAYSCALE) / 255.            
      data, _ = pad_to_next_multiply_of_n(image, 8)
      label, _ = pad_to_next_multiply_of_n(truth, 8)
      return np.expand_dims(data, axis=0), np.expand_dims(label, axis=0)
    else:
      files = self.dataset[(i * self.batch_size):((i + 1) * self.batch_size)]
      data = np.zeros((self.batch_size, self.patch_size[0], self.patch_size[1], 3))
      labels = np.zeros((self.batch_size, self.patch_size[0], self.patch_size[1], 1))

      for i, sample in enumerate(files):
        image_file = sample.split(',')[0]
        truth_file = sample.split(',')[1][:-1]
        image = np.float32(cv2.imread(os.path.join(self.data_location, image_file)) / 255.0)
        truth = cv2.imread(os.path.join(self.data_location, truth_file), cv2.IMREAD_GRAYSCALE) / 255.
        data[i], labels[i] = crop_random(image, truth, size=self.patch_size)
      return data, labels


class UmikryFaceDetector():
  def __init__(self, pretrained_weights=None, encoder_weights=None):
    self.build()

    if pretrained_weights is not None:
      self.model.load_weights(pretrained_weights)
    elif encoder_weights is not None:
      self.model.load_weights(encoder_weights, by_name=True)

  def build(self):
    image = Input(shape=(None, None, 3))
    conv1 = Conv2D(16, 3, padding='same', name='d_conv_1')(image)
    conv1 = LeakyReLU(alpha=0.1, name='d_relu_1')(conv1)
    pool1 = MaxPooling2D(name='d_pool1')(conv1)
    
    conv2 = Conv2D(32, 3, padding='same', use_bias=False, name='d_conv_2')(pool1)
    conv2 = BatchNormalization(momentum=0.8, name='d_bn_1')(conv2)
    conv2 = LeakyReLU(alpha=0.1, name='d_relu_2')(conv2)
    pool2 = MaxPooling2D(name='d_pool2')(conv2)
    
    conv3 = Conv2D(64, 3, padding='same', use_bias=False, name='d_conv_3')(pool2)
    conv3 = BatchNormalization(momentum=0.8, name='d_bn_2')(conv3)
    conv3 = LeakyReLU(alpha=0.1, name='d_relu_3')(conv3)
    pool3 = MaxPooling2D(name='d_pool3')(conv3)

    conv4 = Conv2D(128, 3, padding='same', use_bias=False, name='d_conv_4')(pool3)
    conv4 = BatchNormalization(momentum=0.8, name='d_bn_3')(conv4)
    conv4 = LeakyReLU(alpha=0.1, name='d_relu_4')(conv4)
    pool4 = MaxPooling2D(name='d_pool4')(conv4)
    
    conv5 = Conv2D(256, 3, padding='same', use_bias=False, name='d_conv_5')(pool4)
    conv5 = BatchNormalization(momentum=0.8, name='d_bn_4')(conv5)
    conv5 = LeakyReLU(alpha=0.1, name='d_relu_5')(conv5)
    
    upscale1 = UpSampling2D(interpolation='bilinear')(conv5)
    conv6 = Conv2DTranspose(128, 3, padding='same', use_bias=False)(upscale1)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU(alpha=0.1)(conv6)
    fuse1 = Add()([conv6, conv4])
    
    upscale2 = UpSampling2D(size=4, interpolation='bilinear')(fuse1)
    conv7 = Conv2DTranspose(32, 3, padding='same', use_bias=False)(upscale2)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU(alpha=0.1)(conv7)
    
    upscale3 = UpSampling2D(interpolation='bilinear')(conv7)
    conv8 = Conv2DTranspose(16, 5, padding='same', use_bias=False)(upscale3)
    conv8 = BatchNormalization()(conv8)
    conv8 = LeakyReLU(alpha=0.1)(conv8)
    fuse2 = Add()([conv8, conv1])
    
    final_score = Conv2D(1, 1, activation='sigmoid', padding='same')(fuse2)

    self.model = Model(image, final_score)
    self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse', 'accuracy'])

  def predict(self, image):
    image, border = pad_to_next_multiply_of_n(image[0], 8)
    image = np.float32(image / 255.0)
    prediction = self.model.predict(np.array([image]))[0]
    prediction = prediction[border[0]:(prediction.shape[0] - border[1]),
                            border[2]:(prediction.shape[1] - border[3]), :]
    return prediction.reshape(prediction.shape[0], prediction.shape[1])

  def train(self, train_generator, epochs=10, steps_per_epoch=None, test_generator=None, validation_steps=None, callbacks=None):
    self.model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch,
                             validation_data=test_generator, validation_steps=validation_steps,
                             use_multiprocessing=True, workers=2, callbacks=callbacks)


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

  umikryFaceDetector = UmikryFaceDetector()
  generateTrainTestSet(data_dir, datasets=['OpenImages'])

  train_sequence = ImageSequence(data_dir, batch_size=16, patch_size=(384, 384), is_training=True)
  test_sequence = ImageSequence(data_dir, batch_size=16, patch_size=(384, 384))

  checkpoint_dir = 'models'
  model_checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, 'community_facedetector_weights.{epoch:02d}-{val_loss:.2f}.h5'), save_weights_only=True, save_best_only=True, monitor='val_acc')
  early_stopping = EarlyStopping(patience=5)

  umikryFaceDetector.train(train_sequence, epochs=20,
                           test_generator=test_sequence,
                           callbacks=[model_checkpoint, early_stopping])
  umikryFaceDetector.model.save_weights(os.path.join(checkpoint_dir, 'community_facedetector_weights.h5'))
