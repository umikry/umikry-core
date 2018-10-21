from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add
import numpy as np
import configparser
import os
import random
import sys
import math
import cv2

object_label = {
  'eye': 255,
  'nose': 192,
  'mouth': 128,
  'face': 64
}


def generateTrainTestSet(data_location, datasets, test_size=0.1, override=True):
  if not os.path.isdir(data_location):
    sys.exit(data_location + ' does not exist. Hint: Use data.py to load the datasets.')

  if override:
    if os.path.isfile(os.path.join(data_location, 'train.txt')):
      os.unlink(os.path.join(data_location, 'train.txt'))
    if os.path.isfile(os.path.join(data_location, 'test.txt')):
      os.unlink(os.path.join(data_location, 'test.txt'))

  for dataset in datasets:
    available_labels = os.listdir(os.path.join(data_location, dataset, 'label'))

    for filename in available_labels:
      data = os.path.join(dataset, 'images', filename)
      label = os.path.join(dataset, 'label', filename)
      if random.random() > test_size:
        with open(os.path.join(data_location, 'train.txt'), 'a') as trainlist:
          trainlist.write(data + ',' + label + '\n')
      else:
        with open(os.path.join(data_location, 'test.txt'), 'a') as testlist:
          testlist.write(data + ',' + label + '\n')


def next_multiply_of_64(x):
  return math.ceil(x / 64) * 64


def pad_to_next_multiply_of_64(image):
  diff_y = next_multiply_of_64(image.shape[0]) - image.shape[0]
  diff_x = next_multiply_of_64(image.shape[1]) - image.shape[1]

  border_top = int(diff_y / 2)
  border_bottom = diff_y - border_top
  border_left = int(diff_x / 2)
  border_right = diff_x - border_left

  return cv2.copyMakeBorder(image, border_top, border_bottom, border_left, border_right,
                            cv2.BORDER_CONSTANT, value=0)


def crop_random(image, truth, size=(64, 64)):
    image = pad_to_next_multiply_of_64(image)
    truth = pad_to_next_multiply_of_64(truth)

    if image.shape[0] > size[0]:
        crop_random_y = random.randint(0, image.shape[0] - size[0])
        image = image[crop_random_y:crop_random_y + size[0], :, :]
        truth = truth[crop_random_y:crop_random_y + size[0], :]

    if image.shape[1] > size[1]:
        crop_random_x = random.randint(0, image.shape[1] - size[1])
        image = image[:, crop_random_x:crop_random_x + size[1], :]
        truth = truth[:, crop_random_x:crop_random_x + size[1]]

    return image, truth.reshape(size[0], size[1], 1)


def ImageGenerator(data_location, batch_size=32, is_training=False, class_to_detect='face'):
  if is_training:
    dataset = open(os.path.join(data_location, 'train.txt')).readlines()
  else:
    dataset = open(os.path.join(data_location, 'test.txt')).readlines()

  while True:
    data = np.zeros((batch_size, 64, 64, 3))
    labels = np.zeros((batch_size, 64, 64, 1))
    for i in range(batch_size):
      random_line = random.choice(dataset)
      image_file = random_line.split(',')[0]
      truth_file = random_line.split(',')[1][:-1]
      image = np.float32(cv2.imread(os.path.join(data_location, image_file)) / 255.0)

      truth_mask = cv2.imread(os.path.join(data_location, truth_file), cv2.IMREAD_GRAYSCALE)
      label = np.zeros_like(truth_mask)
      label[truth_mask == object_label[class_to_detect]] = 1

      data[i], labels[i] = crop_random(image, label)
    yield data, labels


class UmikryFaceDetector():
  def __init__(self, pretrained_weights=None):
    self.build()

    if pretrained_weights is not None:
      self.model.load_weights(pretrained_weights)

  def build(self):
    image = Input(shape=(None, None, 3))
    conv1 = Conv2D(8, 3, activation='relu', padding='same')(image)
    pool1 = MaxPooling2D()(conv1)
    conv2 = Conv2D(16, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=4)(conv2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=4)(conv3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(pool3)
    pool4 = MaxPooling2D()(conv4)
    conv5 = Conv2D(256, 8, activation='relu', padding='same')(pool4)
    upscale1 = UpSampling2D()(conv5)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(upscale1)
    fuse1 = Add()([conv6, conv4])
    upscale2 = UpSampling2D(size=4)(fuse1)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(upscale2)
    upscale3 = UpSampling2D(size=4)(conv7)
    conv8 = Conv2D(16, 3, activation='relu', padding='same')(upscale3)
    fuse2 = Add()([conv8, conv2])
    upscale4 = UpSampling2D()(fuse2)
    conv9 = Conv2D(8, 3, activation='relu', padding='same')(upscale4)
    score = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9)

    self.model = Model(inputs=image, outputs=score)
    self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse', 'accuracy'])

  def predict(self, image):
    prediction = self.model.predict(np.array([image]))[0]
    return prediction.reshape(prediction.shape[0], prediction.shape[1])

  def train(self, train_generator, epochs=10, steps_per_epoch=1000, test_generator=None, validation_steps=None):
    self.model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch,
                             validation_data=test_generator, validation_steps=validation_steps)


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

  train_generator = ImageGenerator(data_dir, is_training=True)
  test_generator = ImageGenerator(data_dir)
  umikryFaceDetector.train(train_generator, epochs=20, steps_per_epoch=1000,
                           test_generator=test_generator, validation_steps=100)
  umikryFaceDetector.model.save(os.path.join('models', 'community_facedetector.h5'))
