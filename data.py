import os
import wget
import zipfile


def downloadCocoData():
  if not os.path.isdir('data'):
    os.mkdir('data')

  if not os.path.isdir('data/coco_val2017'):
    os.makedirs('data/coco/val2017')
    wget.download('http://images.cocodataset.org/zips/val2017.zip', 'data/coco/val2017.zip')

    with zipfile.ZipFile('data/coco/val2017.zip', 'r') as archive:
      archive.extractall('data/coco/val2017')

    os.unlink('data/coco/val2017.zip')
  else:
    print('Please remove the existing coco folder to proceed!')


if __name__ == '__main__':
  downloadCocoData()
