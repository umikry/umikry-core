import os
import wget
import zipfile
import sys


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
      number_of_lines = len(open(meta_dir + 'validation-images-with-rotation.csv').readlines())
      for i, line in enumerate(lines):
        sys.stdout.write("\r{}/{}".format(i, number_of_lines))
        sys.stdout.write("\033[K")
        sys.stdout.flush()
        image_id, _, original_url = line.split(',')[:3]
        if image_id in all_image_ids:
          wget.download(original_url, image_dir + original_url.split('/')[-1], bar=None)
          # TODO: clear flickr no longer available images
          # TODO: draw truth out of bounding boxes
          # TODO: save subimages like nose etc.


if __name__ == '__main__':
  # downloadCocoData()
  downloadOpenImagesData()
