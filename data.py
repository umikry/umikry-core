import os
import wget
import zipfile
import sys
import requests


def downloadCocoData():
  if not os.path.isdir('data/coco/val2017'):
    os.makedirs('data/coco/val2017')
    wget.download('http://images.cocodataset.org/zips/val2017.zip', 'data/coco/val2017.zip')

    with zipfile.ZipFile('data/coco/val2017.zip', 'r') as archive:
      archive.extractall('data/coco')

    os.unlink('data/coco/val2017.zip')
  else:
    print('Please remove the existing coco folder to proceed!')

  # TODO: Extract only humans/people/faces/eyes/ ... etc


def downloadOpenImagesData():
  if not os.path.isdir('data/OpenImages'):
    data_dir = 'data/OpenImages/'
    image_dir = data_dir + 'images/'
    meta_dir = data_dir + 'meta/'
    os.makedirs(meta_dir)
    os.mkdir(image_dir)

    openimages_url = 'https://storage.googleapis.com/openimages/2018_04/'

    wget.download(openimages_url + 'validation/validation-annotations-bbox.csv',
                  meta_dir + 'validation-annotations-bbox.csv')

    wget.download(openimages_url + 'validation/validation-images-with-rotation.csv',
                  meta_dir + 'validation-images-with-rotation.csv')

    classes = [('human_eye', '/m/014sv8'), ('human_nose', '/m/0k0pj'),
               ('human_mouth', '/m/0283dt1'), ('human_face', '/m/0dzct')]

    all_image_ids = set()
    for object_class in classes:
      os.mkdir('data/OpenImages/' + object_class[0])
      image_ids = set()
      with open(meta_dir + 'validation-annotations-bbox.csv', 'r') as lines:
        for line in lines:
          image_id, _, label_name, _, box_x_start, box_x_end, box_y_start, box_y_end = line.split(',')[:8]
          if label_name == object_class[1]:
            image_ids.add(image_id)
        all_image_ids = all_image_ids | image_ids

    with open(meta_dir + 'validation-images-with-rotation.csv', 'r') as lines:
      summary = {
        'total_images': len(open(meta_dir + 'validation-images-with-rotation.csv').readlines()),
        'images_in_classlist': 0,
        'not_available': 0
      }
      for i, line in enumerate(lines):
        sys.stdout.write("\r{}/{}".format(i, summary['total_images']))
        sys.stdout.write("\033[K")
        sys.stdout.flush()
        image_id, _, original_url = line.split(',')[:3]
        if image_id in all_image_ids:
          summary['images_in_classlist'] = summary['images_in_classlist'] + 1
          url = original_url
          file_name = original_url.split('/')[-1]
          req = requests.get(url)
          # check if filename changed eg. due to a flickr no longer available image
          if req.url.split('/')[-1] == file_name:
            with open(image_dir + file_name, 'wb') as image:
              image.write(req.content)
          else:
            summary['not_available'] = summary['not_available'] + 1
          # TODO: draw truth out of bounding boxes
          # TODO: save subimages like nose etc.
    sys.stdout.write("\r")
    sys.stdout.write("\033[K")
    sys.stdout.flush()
    print(('{} pictures were downloaded. {} OpenImages (val) total,'
          ' {} in class list and {} are not available anymore').format(
          summary['images_in_classlist'] - summary['not_available'],
          summary['total_images'], summary['images_in_classlist'],
          summary['not_available']))


if __name__ == '__main__':
  # downloadCocoData()
  downloadOpenImagesData()
