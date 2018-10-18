import os
import wget
import zipfile


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
    meta_dir = 'data/OpenImages/meta/'
    os.makedirs(meta_dir)

    openimages_url = 'https://storage.googleapis.com/openimages/2018_04/'

    wget.download(openimages_url + 'validation/validation-images-with-rotation.csv',
                  meta_dir + 'validation-images-with-rotation.csv')

    wget.download(openimages_url + 'validation/validation-annotations-human-imagelabels.csv',
                  meta_dir + 'validation-annotations-human-imagelabels.csv')

    classes = [('Human_eye', '/m/014sv8'), ('Human_nose', '/m/0k0pj'),
               ('Human_mouth', '/m/0283dt1'), ('Human_face', '/m/0dzct')]

    for object_class in classes:
      os.makedir('data/OpenImages/' + object_class[0])
      image_ids = {}
      with open(meta_dir + 'validation-annotations-human-imagelabels.csv', 'r') as lines:
        for line in lines:
          image_id, _, label_name, _ = line.split(',')
          if label_name == object_class:
            image_ids |= label_name
            # TODO Download all important images and labels


if __name__ == '__main__':
  downloadCocoData()
  downloadOpenImagesData()
