import bz2
import dlib
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import wget
import cv2


class UmikryFaceRecognizer(object):

    def __init__(self, shape_predictor='SMALL', threshold=0.6):
        self.shape_predictor = shape_predictor
        self.threshold = threshold

        if self.shape_predictor == 'SMALL':
            shape_predictor_path = os.path.join('models', 'shape_predictor_5_face_landmarks.dat')
            if not os.path.isfile(shape_predictor_path):
                shape_predictor_url = 'http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2'
                shape_predictor_path_bz = shape_predictor_path + '.bz2'
                wget.download(shape_predictor_url, shape_predictor_path_bz)
                self.__decompress_bz2_file(shape_predictor_path, shape_predictor_path_bz)
            self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
        elif self.shape_predictor == 'BIG':
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

    def recognize_faces(self, image, faces):
        # TODO: Allow multiple vectors per person
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
                    person = input("This person is not in the database. Please add this person or hit enter to conitue. Name: ")
                    max_id = max(self.database, key=int) + 1
                    if not person:
                        person = "Unknown"
                    self.database[max_id] = [person, encoded_face]
                    joblib.dump(self.database, self.database_path, compress=3)
