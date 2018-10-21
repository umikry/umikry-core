from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential
import numpy as np
import configparser
import os
import random
import sys
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


def random_crop_or_pad(image, truth, size=(64, 64)):
    assert image.shape[:2] == truth.shape[:2]

    if image.shape[0] > size[0]:
        crop_random_y = random.randint(0, image.shape[0] - size[0])
        image = image[crop_random_y:crop_random_y + size[0], :, :]
        truth = truth[crop_random_y:crop_random_y + size[0], :]
    else:
        zeros = np.zeros((size[0], image.shape[1], image.shape[2]), dtype=np.float32)
        zeros[:image.shape[0], :image.shape[1], :] = image
        image = np.copy(zeros)
        zeros = np.zeros((size[0], truth.shape[1]), dtype=np.float32)
        zeros[:truth.shape[0], :truth.shape[1]] = truth
        truth = np.copy(zeros)

    if image.shape[1] > size[1]:
        crop_random_x = random.randint(0, image.shape[1] - size[1])
        image = image[:, crop_random_x:crop_random_x + size[1], :]
        truth = truth[:, crop_random_x:crop_random_x + size[1]]
    else:
        zeros = np.zeros((image.shape[0], size[1], image.shape[2]))
        zeros[:image.shape[0], :image.shape[1], :] = image
        image = np.copy(zeros)
        zeros = np.zeros((truth.shape[0], size[1]))
        zeros[:truth.shape[0], :truth.shape[1]] = truth
        truth = np.copy(zeros)

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

      data[i], labels[i] = random_crop_or_pad(image, label)
    yield data, labels


class UmikryFaceDetector():
  def __init__(self):
    self.build()

  def build(self):
    self.model = Sequential()
    self.model.add(Conv2D(16, 3, activation='relu', padding='same', input_shape=(None, None, 3)))
    self.model.add(MaxPooling2D())
    self.model.add(Conv2D(32, 3, activation='relu', padding='same'))
    self.model.add(MaxPooling2D())
    self.model.add(Conv2D(64, 3, activation='relu', padding='same'))
    self.model.add(Conv2D(1, 1, activation='sigmoid', padding='same'))
    self.model.add(UpSampling2D())
    self.model.add(Conv2D(1, 3, activation='relu', padding='same'))
    self.model.add(UpSampling2D())
    self.model.add(Conv2D(1, 1, activation='sigmoid', padding='same'))

    self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse', 'accuracy'])

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
  umikryFaceDetector.train(train_generator, epochs=10, steps_per_epoch=500,
                           test_generator=test_generator, validation_steps=50)
  umikryFaceDetector.model.save(os.path.join('models', 'community_facedetector.h5'))
