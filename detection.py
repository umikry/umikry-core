from keras.layers import Conv2d, MaxPooling2D, UpSampling2D
from keras.model import Sequential
import numpy as np


def ImageGenerator(batch_size=32, is_training=False):
  while True:
    data = np.zeros((batch_size, 256, 256, 3))
    labels = np.zeros((batch_size, 256, 256, 1))

    yield data, labels


model = Sequential()
model.add(Conv2d(32, 3, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Conv2d(64, 3, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Conv2d(128, 3, activation='relu', padding='same'))
model.add(Conv2d(1, 1, activation='sigmoid', padding='same'))
model.add(UpSampling2D())
model.add(Conv2d(1, 3, activation='relu', padding='same'))
model.add(UpSampling2D())
model.add(Conv2d(1, 1, activation='sigmoid', padding='same'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse', 'accuracy'])

train_generator = ImageGenerator(is_training=True)
test_generator = ImageGenerator()

model.fit_generator(train_generator, epochs=10, steps_per_epoch=1000,
                    validation_data=test_generator, validation_steps=100)
