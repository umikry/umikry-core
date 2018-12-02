'''
Copyright (c) 2018, umikry.com
License AGPL-3.0

This code is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License, version 3,
as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License, version 3,
along with this program.  If not, see <http://www.gnu.org/licenses/>
'''

import os
import wget
import zipfile
import sys
import requests
import cv2
import configparser
import numpy as np
import shutil
from google_drive_downloader import GoogleDriveDownloader as gdd


class UmikryDataPioneer():
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def prepareWIDERFaceForHaarTraining(self, override=False):
        data_dir = os.path.join(self.base_dir, 'WIDERFace')
        if os.path.isdir(data_dir):
            train_dir = os.path.join(self.base_dir, 'WIDERFace_Haar')
            if override and os.path.isdir(train_dir):
                shutil.rmtree(train_dir)

            if not os.path.isdir(train_dir):
                positive_dir = os.path.join(train_dir, 'positives')
                negatives_dir = os.path.join(train_dir, 'negatives')
                os.makedirs(positive_dir)
                os.mkdir(negatives_dir)
                label_file = os.path.join(data_dir, 'wider_face_split', 'wider_face_train_bbx_gt.txt')
                with open(label_file, 'r') as lines:
                    filename = None
                    image = None
                    face_removed_image = None
                    boundingBoxes = 0
                    for line in lines:
                        if line[:-1].endswith('jpg'):
                            if face_removed_image is not None:
                                cv2.imwrite(os.path.join(negatives_dir, filename.split('/')[1]), face_removed_image)
                            filename = line[:-1]
                            image = cv2.imread(os.path.join(data_dir, 'WIDER_train', 'images', filename))
                            face_removed_image = image.copy()
                        elif boundingBoxes < 1:
                            boundingBoxes = int(line[:-1])
                        else:
                            x, y, w, h, blur, _, illumination, invalid, occlusion, _ = line[:-2].split(' ')
                            if image is not None:
                                if int(w) > 50 and int(h) > 50 and blur == '0' and illumination == '0' and invalid == '0' and occlusion == '0':
                                    save_path = os.path.join(positive_dir, str(boundingBoxes) + '_' + filename.split('/')[1])
                                    cv2.imwrite(save_path,
                                                image[int(y):int(y) + int(h), int(x):int(x) + int(w)])
                                face_removed_image[int(y):int(y) + int(h), int(x):int(x) + int(w)] = 0
                            boundingBoxes = boundingBoxes - 1

            else:
                print('Skip haar training set generation: {} already exists!'.format(train_dir))
        else:
            print('{} does not exist. Please use a UmikryDataCollector to fetch it first.'.format(data_dir))


class UmikryDataCollector(object):
    def __init__(self, base_dir, verbose=False):
        self.base_dir = base_dir
        self.verbose = verbose

    def __info(self, message):
        if self.verbose:
            print(message)

    def download_coco_data(self):
        data_dir = os.path.join(self.base_dir, 'coco')
        if not os.path.isdir(data_dir):
            os.makedirs(os.path.join(data_dir))
            wget.download('http://images.cocodataset.org/zips/val2017.zip', os.path.join(data_dir, 'val2017.zip'))

            with zipfile.ZipFile(os.path.join(data_dir, 'val2017.zip'), 'r') as archive:
                archive.extractall(data_dir)

            os.unlink(os.path.join(data_dir, 'val2017.zip'))
        else:
            self.__info('Please remove the existing coco folder to proceed!')

    def downloadWIDERFaceDataset(self):
        acknowledgement = """WIDER FACE
        A Face Detection Benchmark
        Multimedia Laboratory, Department of Information Engineering,
        The Chinese University of Hong Kong

        http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/"""
        self.__info('Note: {}'.format(acknowledgement))

        data_dir = os.path.join(self.base_dir, 'WIDERFace')
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

            self.__info('The download of WIDERFace might take a while (~ 10 min @ 32 Mbit/s)')
            if not os.path.exists(os.path.join(data_dir, 'Caltech_WebFaces_train.zip')):
                gdd.download_file_from_google_drive(file_id='0B6eKvaijfFUDQUUwd21EckhUbWs',
                                                    dest_path=os.path.join(data_dir, 'Caltech_WebFaces_train.zip'),
                                                    unzip=False)
            else:
                self.__info('Skip download because Caltech_WebFaces_train.zip still exists')

            with zipfile.ZipFile(os.path.join(data_dir, 'Caltech_WebFaces_train.zip'), 'r') as archive:
                archive.extractall(data_dir)

            os.remove(os.path.join(data_dir, 'Caltech_WebFaces_train.zip'))

            annotations_url = 'http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip'
            wget.download(annotations_url, os.path.join(data_dir, 'wider_face_split.zip'))

            with zipfile.ZipFile(os.path.join(data_dir, 'wider_face_split.zip'), 'r') as archive:
                archive.extractall(os.path.join(data_dir))
        else:
            self.__info('Skip WIDERFace download: {} already exists!'.format(data_dir))

    def download_open_images_data(self):
        data_dir = os.path.join(self.base_dir, 'OpenImages')
        if not os.path.isdir(data_dir):
            image_dir = os.path.join(data_dir, 'images')
            label_dir = os.path.join(data_dir, 'label')
            face_dir = os.path.join(data_dir, 'faces_center_cropped')
            face_label_dir = os.path.join(data_dir, 'faces_center_cropped_label')
            meta_dir = os.path.join(data_dir, 'meta')
            os.makedirs(data_dir)
            os.mkdir(meta_dir)
            os.mkdir(face_dir)
            os.mkdir(face_label_dir)
            os.mkdir(image_dir)
            os.mkdir(label_dir)

            openimages_url = 'https://storage.googleapis.com/openimages/2018_04/'

            wget.download(openimages_url + 'validation/validation-annotations-bbox.csv',
                          os.path.join(meta_dir, 'validation-annotations-bbox.csv'))

            wget.download(openimages_url + 'validation/validation-images-with-rotation.csv',
                          os.path.join(meta_dir, 'validation-images-with-rotation.csv'))

            object_classes = {
                '/m/014sv8': 'human_eye',
                '/m/0k0pj': 'human_nose',
                '/m/0283dt1': 'human_mouth',
                '/m/0dzct': 'human_face'
            }

            object_label = {
                'human_eye': 255,
                'human_nose': 192,
                'human_mouth': 128,
                'human_face': 64
            }

            for object_class in object_classes.items():
                os.mkdir(os.path.join(data_dir, object_class[1]))

            image_meta_data = []
            with open(os.path.join(meta_dir, 'validation-annotations-bbox.csv'), 'r') as lines:
                for line in lines:
                    image_id, _, label_name, _, box_x_start, box_x_end, box_y_start, box_y_end = line.split(',')[:8]
                    if label_name in [object_class[0] for object_class in object_classes.items()]:
                        image_meta_data.append((image_id, object_classes[label_name],
                                                float(box_x_start), float(box_x_end),
                                                float(box_y_start), float(box_y_end)))

            distinct_image_ids = set([meta_data[0] for meta_data in image_meta_data])
            with open(os.path.join(meta_dir, 'validation-images-with-rotation.csv'), 'r', encoding='utf8') as lines:
                summary = {
                    'total_images': len(
                        open(os.path.join(meta_dir, 'validation-images-with-rotation.csv'), encoding='utf8').readlines()),
                    'images_in_classlist': 0,
                    'not_available': 0
                }
                for i, line in enumerate(lines):
                    sys.stdout.write("\r{}/{}".format(i, summary['total_images']))
                    sys.stdout.write("\033[K")
                    sys.stdout.flush()
                    image_id, _, original_url = line.split(',')[:3]
                    if image_id in distinct_image_ids:
                        summary['images_in_classlist'] = summary['images_in_classlist'] + 1
                        url = original_url
                        file_name = original_url.split('/')[-1]
                        try:
                            req = requests.get(url)
                        except Exception as e:
                            print('Error while opening ' + original_url)
                            print(e)
                            break

                        # check if filename changed eg. due to a flickr no longer available image
                        if req.url.split('/')[-1] == file_name:
                            try:
                                with open(os.path.join(image_dir, file_name), 'wb') as image_file:
                                    image_file.write(req.content)
                            except Exception as e:
                                print('Error while fetching ' + original_url)
                                print(e)
                                break
                            events = [meta_data for meta_data in image_meta_data if meta_data[0] == image_id]
                            image = cv2.imread(os.path.join(image_dir, file_name))
                            if image is None:
                                print('\n' + file_name + ' seems to be empty')
                            else:
                                height = image.shape[0]
                                width = image.shape[1]
                                label = np.zeros((height, width))
                                face_label = np.zeros((image.shape[0], image.shape[1]))
                                faces = []
                                for j, event in enumerate(events):
                                    y_start = int(event[4] * height)
                                    y_end = int(event[5] * height)
                                    x_start = int(event[2] * width)
                                    x_end = int(event[3] * width)

                                    if event[1] == 'human_face':
                                        border = 100
                                        border_y_start = (y_start - border) if (y_start - border) > 0 else 0
                                        border_y_end = (y_end + border) if (y_end + border) < image.shape[0] else image.shape[0]
                                        border_x_start = (x_start - border) if (x_start - border) > 0 else 0
                                        border_x_end = (x_end + border) if (x_end + border) < image.shape[1] else image.shape[1]
                                        faces.append((border_y_start, border_y_end, border_x_start, border_x_end))
                                        face_label[y_start:y_end, x_start:x_end] = 255

                                    cv2.imwrite(os.path.join(data_dir, event[1], '{}_{}'.format(j, file_name)),
                                                image[y_start:y_end, x_start:x_end])
                                    # set label color, but do not override other labels
                                    label[y_start:y_end, x_start:x_end][np.where(
                                        label[y_start:y_end, x_start:x_end] == 0)] = object_label[event[1]]

                                cv2.imwrite(os.path.join(label_dir, file_name), label)
                                if faces:
                                    for j, face in enumerate(faces):
                                        cv2.imwrite(os.path.join(face_dir, '{}_{}'.format(j, file_name)),
                                                    image[face[0]:face[1], face[2]:face[3]])
                                        cv2.imwrite(os.path.join(face_label_dir, '{}_{}'.format(j, file_name)),
                                                    face_label[face[0]:face[1], face[2]:face[3]])
                        else:
                            summary['not_available'] = summary['not_available'] + 1
            sys.stdout.write("\r")
            sys.stdout.write("\033[K")
            sys.stdout.flush()
            self.__info(('{} pictures were downloaded. {} OpenImages (val) total,'
                       '{} in class list and {} are not available anymore').format(
                summary['images_in_classlist'] - summary['not_available'],
                summary['total_images'], summary['images_in_classlist'],
                summary['not_available']))
        else:
            print('OpenImages folder already exists. Please remove it manually to proceed.')


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('umikry.ini')
    if 'DATA' in config:
        data_dir = config['DATA']['Location']
    else:
        data_dir = input('Choose a directory to store the data:')
        config['DATA'] = {'Location': data_dir}

        with open('umikry.ini', 'w') as configfile:
            config.write(configfile)

    umikryDataCollector = UmikryDataCollector(base_dir=data_dir, verbose=True)
    umikryDataPioneer = UmikryDataPioneer(base_dir=data_dir)
    umikryDataCollector.downloadWIDERFaceDataset()
    umikryDataPioneer.prepareWIDERFaceForHaarTraining(override=True)
