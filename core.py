from detection import UmikryFaceDetector
from transformation import UmikryFaceTransformator


def umikry(image, detection='caffe', transformation='blur'):
    umikryFaceDetector = UmikryFaceDetector(method=detection)
    umikryFaceTransformator = UmikryFaceTransformator(method=transformation)

    faces = umikryFaceDetector.detect(image)
    image = umikryFaceTransformator.transform(image, faces)

    return image
