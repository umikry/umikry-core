from keras.layers import Conv2d, MaxPooling2D, UpSampling2D
from keras.model import Sequential
import numpy as np
import configparser
import os


def ImageGenerator(datasets, data_location, batch_size=32, is_training=False):
  data_locations = []
  for dataset in datasets:
    data_locations.append(os.path.join(data_dir, 'OpenImages'))

  os.path.join(data_dir, 'OpenImages')

  while True:
    data = np.zeros((batch_size, 256, 256, 3))
    labels = np.zeros((batch_size, 256, 256, 1))

    yield data, labels


class UmikryFaceDetector():
  def __init__(self):
    self.build()

  def build(self):
    self.model = Sequential()
    self.model.add(Conv2d(32, 3, activation='relu', padding='same'))
    self.model.add(MaxPooling2D())
    self.model.add(Conv2d(64, 3, activation='relu', padding='same'))
    self.model.add(MaxPooling2D())
    self.model.add(Conv2d(128, 3, activation='relu', padding='same'))
    self.model.add(Conv2d(1, 1, activation='sigmoid', padding='same'))
    self.model.add(UpSampling2D())
    self.model.add(Conv2d(1, 3, activation='relu', padding='same'))
    self.model.add(UpSampling2D())
    self.model.add(Conv2d(1, 1, activation='sigmoid', padding='same'))

    self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse', 'accuracy'])

  def train(self, train_generator, epochs=10, steps_per_epoch=1000, validation_data=None, validation_steps=None):
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
  train_generator = ImageGenerator(is_training=True, data_location=data_dir, datasets=['OpenImages'])
  test_generator = ImageGenerator(data_location=data_dir, datasets=['OpenImages'])
  umikryFaceDetector.train()
