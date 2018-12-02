import bz2
import dlib
import numpy as np
import os
import wget


class UmikryFaceRecognizer():

    def __init__(self, shape_predictor='SMALL', threshold=0.6):
        self.shape_predictor = shape_predictor
        self.threshold = threshold
        root_dir = os.path.dirname(os.path.realpath(__file__))

        if self.shape_predictor == 'SMALL':
            shape_predictor_path = os.path.join(root_dir, 'models', 'shape_predictor_5_face_landmarks.dat')
            if not os.path.isfile(shape_predictor_path):
                shape_predictor_url = 'http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2'
                shape_predictor_path_bz = shape_predictor_path + '.bz2'
                wget.download(shape_predictor_url, shape_predictor_path_bz)
                self.__decompress_bz2_file(shape_predictor_path, shape_predictor_path_bz)
            self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
        elif self.shape_predictor == 'BIG':
            shape_predictor_path = os.path.join(root_dir, 'models', 'shape_predictor_68_face_landmarks.dat')
            if not os.path.isfile(shape_predictor_path):
                shape_predictor_url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
                shape_predictor_path_bz = shape_predictor_path + '.bz2'
                wget.download(shape_predictor_url, shape_predictor_path_bz)
                self.__decompress_bz2_file(shape_predictor_path, shape_predictor_path_bz)
            self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

        face_rec_path = os.path.join(root_dir, 'models', 'dlib_face_recognition_resnet_model_v1.dat')
        if not os.path.isfile(face_rec_path):
            face_rec_url = 'http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2'
            face_rec_path_bz = face_rec_path + '.bz2'
            wget.download(face_rec_url, face_rec_path_bz)
            self.__decompress_bz2_file(face_rec_path, face_rec_path_bz)
        self.face_rec = dlib.face_recognition_model_v1(face_rec_path)

    def __decompress_bz2_file(self, file_path, path_bz2):
        zipfile = bz2.BZ2File(path_bz2)

        with open(file_path, 'wb') as file:
            file.write(zipfile.read())
            file.close()

        zipfile.close()
        os.remove(path_bz2)

    def recognize_faces(self, image, faces, familiar_faces):
        recognized_faces = {}

        for i, (left, top, right, bottom) in faces.items():
            rect = dlib.rectangle(left, top, right, bottom)
            shape = self.shape_predictor(image, rect)
            encoded_face = np.array(self.face_rec.compute_face_descriptor(image, shape))

            for _, value in familiar_faces.items():
                score = np.linalg.norm(value - encoded_face)
                recognized_faces[i] = {
                    "bounding_box": (left, top, right, bottom),
                    "is_familiar": score <= self.threshold
                }

        return recognized_faces
