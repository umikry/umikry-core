import os
import wget
import zipfile
import sys
import requests
import cv2
import configparser
import numpy as np


def download_coco_data(base_dir=''):
    if not os.path.isdir('data/coco/val2017'):
        os.makedirs('data/coco/val2017')
        wget.download('http://images.cocodataset.org/zips/val2017.zip', 'data/coco/val2017.zip')

        with zipfile.ZipFile('data/coco/val2017.zip', 'r') as archive:
            archive.extractall('data/coco')

        os.unlink('data/coco/val2017.zip')
    else:
        print('Please remove the existing coco folder to proceed!')

    # TODO: Extract humans/people/faces/eyes/ ... etc like OpenImages dataset


def download_open_images_data(base_dir=''):
    if not os.path.isdir(os.path.join(base_dir, 'OpenImages')):
        data_dir = os.path.join(base_dir, 'OpenImages')
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
                                    border_y_end = (y_end + border) if (y_end + border) < image.shape[0] else \
                                    image.shape[0]
                                    border_x_start = (x_start - border) if (x_start - border) > 0 else 0
                                    border_x_end = (x_end + border) if (x_end + border) < image.shape[1] else \
                                    image.shape[1]
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
        print(('{} pictures were downloaded. {} OpenImages (val) total,'
               ' {} in class list and {} are not available anymore').format(
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
    # download_coco_data()
    download_open_images_data(base_dir=data_dir)
