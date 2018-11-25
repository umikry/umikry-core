from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add, BatchNormalization, LeakyReLU, Conv2DTranspose
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import Sequence
from sklearn.externals import joblib
import numpy as np
import os
import dlib
import random
import sys
import math
import cv2
import wget
import bz2
import matplotlib.pyplot as plt

def generate_train_test_set(data_location, datasets, test_size=0.1, override=True):
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


class UmikryFaceDetector(object):
    def __init__(self, method='haar', pretrained_weights=None, encoder_weights=None):
        self.method = method

        if self.method == 'cnn':
            self.__build()
            if pretrained_weights is not None:
                self.model.load_weights(pretrained_weights)
            elif encoder_weights is not None:
                self.model.load_weights(encoder_weights, by_name=True)
        elif self.method == 'haar':
            haar_classifier_path = os.path.join('models', 'haarcascade_frontalface_default.xml')
            if not os.path.isfile(haar_classifier_path):
                haar_classifier_url = 'https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml'
                wget.download(haar_classifier_url, haar_classifier_path)

            self.haar_classifier = cv2.CascadeClassifier(haar_classifier_path)
        elif self.method == 'caffe':
            caffe_model_path = os.path.join('models', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
            caffe_config_path = os.path.join('models', 'deploy.prototxt')
            if not os.path.isfile(caffe_model_path):
                caffe_model_url = 'https://github.com/opencv/opencv_3rdparty/raw/19512576c112aa2c7b6328cb0e8d589a4a90a26d/res10_300x300_ssd_iter_140000_fp16.caffemodel'
                wget.download(caffe_model_url, caffe_model_path)
            if not os.path.isfile(caffe_config_path):
                caffe_config_url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt'
                wget.download(caffe_config_url, caffe_config_path)
            self.caffe_classifier = cv2.dnn.readNetFromCaffe(caffe_config_path, caffe_model_path)

        else:
            raise ValueError('{} is an invalid method use \'cnn\' or \'haar\' instead'.format(self.method))

    def __build(self):
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

    def __haar_detection(self, image, scale_factor=1.1, min_neighbors=3):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.haar_classifier.detectMultiScale(image)

        if len(faces) > 0:
            faces_dict = {}
            for i, (x, y, width, height) in enumerate(faces):
                faces_dict[i] = (x, y, x + width, y + height)
            return faces_dict
        else:
            return None

    def __caffe_detection(self, image):
        (h, w) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 0.4, (w, h), [104, 117, 123], False, False)
        self.caffe_classifier.setInput(blob)
        faces = self.caffe_classifier.forward()
        if np.any(faces[0, 0, :, 2] > 0.4):  # threshold
            faces_dict = {}
            for i in range(0, faces.shape[2]):
                if faces[0, 0, i, 2] > 0.4:  # faces[0, 0, i, 2] --> confidence of net
                    box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x, y, x_w, y_h = box.astype("int")
                    faces_dict[i] = [x, y, x_w, y_h]
            return faces_dict
        else:
            return None

    def detect(self, image, scale_factor=1.1, min_neighbors=3):
        if self.method == 'haar':
            return self.__haar_detection(image)
        elif self.method == 'caffe':
            return self.__caffe_detection(image)
        elif self.method == 'cnn':
            image, border = pad_to_next_multiply_of_n(image[0], 8)
            image = np.float32(image / 255.0)
            prediction = self.model.predict(np.array([image]))[0]
            prediction = prediction[border[0]:(prediction.shape[0] - border[1]),
                         border[2]:(prediction.shape[1] - border[3]), :]
            return prediction.reshape(prediction.shape[0], prediction.shape[1])

    def __haar_train(self, train_folder):
        if (not os.path.isdir(os.path.join(train_folder, 'positives')) or
                not os.path.isdir(os.path.join(train_folder, 'negatives'))):
            print('train_folder needs to include positives and negatives subfolders')
            print('Hint: Use UmikryDataPioneer to generate train data')
        else:
            raise NotImplementedError

    def train(self, train_generator, epochs=10, steps_per_epoch=None, test_generator=None,
              validation_steps=None, callbacks=None, train_folder=None):
        if self.method == 'haar':
            self.__haar_train(train_folder)
        if self.method == 'caffe':
            raise NotImplementedError
        elif self.method == 'cnn':
            self.model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch,
                                     validation_data=test_generator, validation_steps=validation_steps,
                                     use_multiprocessing=True, workers=2, callbacks=callbacks)


class UmikryFaceRecognizer(object):

    def __init__(self, detect_method='caffe', shape_predictor='small', threshold=0.6):
        self.face_detector = UmikryFaceDetector(method=detect_method)
        self.shape_predictor = shape_predictor
        self.threshold = threshold

        if self.shape_predictor == 'small':
            shape_predictor_path = os.path.join('models', 'shape_predictor_5_face_landmarks.dat')
            if not os.path.isfile(shape_predictor_path):
                shape_predictor_url = 'http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2'
                shape_predictor_path_bz = shape_predictor_path + '.bz2'
                wget.download(shape_predictor_url, shape_predictor_path_bz)
                self.__decompress_bz2_file(shape_predictor_path, shape_predictor_path_bz)
            self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
        elif self.shape_predictor == 'big':
            shape_predictor_path = os.path.join('models', 'shape_predictor_68_face_landmarks.dat')
            if not os.path.isfile(shape_predictor_path):
                shape_predictor_url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
                shape_predictor_path_bz = shape_predictor_path + '.bz2'
                wget.download(shape_predictor_url, shape_predictor_path_bz)
                self.__decompress_bz2_file(shape_predictor_path, shape_predictor_path_bz)
            self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

        face_rec_path = os.path.join('models', 'dlib_face_recognition_resnet_model_v1.dat')
        if not os.path.isfile(face_rec_path):
            face_rec_url = 'http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2'
            face_rec_path_bz = face_rec_path + '.bz2'
            wget.download(face_rec_url, face_rec_path_bz)
            self.__decompress_bz2_file(face_rec_path, face_rec_path_bz)
        self.face_rec = dlib.face_recognition_model_v1(face_rec_path)

        self.database_path = os.path.join('models', 'database.sav')
        if not os.path.isfile(self.database_path):
            database = {}
            joblib.dump(database, self.database_path, compress=3)
        else:
            self.database = joblib.load(self.database_path)

    def __decompress_bz2_file(self, path, path_bz2):
        zipfile = bz2.BZ2File(path_bz2)
        f = open(path, 'wb')
        f.write(zipfile.read())
        f.close()
        zipfile.close()
        os.remove(path_bz2)

    def __show_image(self, image, rect, person=None):
        image_copy = image.copy()
        cv2.rectangle(image_copy, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 0, 255), 2)
        if person is not None:
            # TODO: resize font scale according to image size
            cv2.putText(image_copy, person, (rect.left() + 6, rect.bottom() - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                        (255, 255, 255), 1)
        plt.figure()
        plt.imshow(image_copy[..., ::-1])
        plt.show()

    def recognize_faces(self, image):
        # TODO: Allow multiple vectors per person
        faces = self.face_detector.detect(image)
        if faces is not None:
            for _, (left, top, right, bottom) in faces.items():
                rect = dlib.rectangle(left, top, right, bottom)
                shape = self.shape_predictor(image, rect)
                encoded_face = np.array(self.face_rec.compute_face_descriptor(image, shape))
                self.database = joblib.load(self.database_path)
                if not self.database:
                    self.__show_image(image, rect)
                    person = input("Database is empty. Please identify the person or hit enter to continue. Name:")
                    if not person:
                        person = "Unknown"
                    self.database[0] = [person, encoded_face]
                    joblib.dump(self.database, self.database_path, compress=3)
                else:
                    for key, values in self.database.items():
                        res = np.linalg.norm(values[1] - encoded_face)
                        if res <= self.threshold:
                            temp_key = key
                            break
                        else:
                            temp_key = None
                    if temp_key is not None:
                        self.__show_image(image, rect, self.database[temp_key][0])
                    else:
                        self.__show_image(image, rect)
                        person = input(
                            "This person is not in the database. Please add this person or hit enter to conitue. Name: ")
                        max_id = max(self.database, key=int) + 1
                        if not person:
                            person = "Unknown"
                        self.database[max_id] = [person, encoded_face]
                        joblib.dump(self.database, self.database_path, compress=3)
        else:
            print("No faces found.")






