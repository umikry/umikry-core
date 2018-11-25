from UmikryFaceDetector import UmikryFaceDetector
from UmikryFaceTransformator import UmikryFaceTransformator


def umikry(image, detection='CAFFE', transformation='GAN'):
    umikryFaceDetector = UmikryFaceDetector(method=detection)
    umikryFaceTransformator = UmikryFaceTransformator(method=transformation)

    faces = umikryFaceDetector.detect(image)
    image = umikryFaceTransformator.transform(image, faces)

    return image
