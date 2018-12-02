from UmikryFaceDetector import UmikryFaceDetector
from UmikryFaceTransformator import UmikryFaceTransformator
from UmikryFaceRecognizer import UmikryFaceRecognizer


def umikry(image, detection='CAFFE', transformation='GAN', familar_faces=None):
    umikryFaceDetector = UmikryFaceDetector(method=detection)
    umikryFaceTransformator = UmikryFaceTransformator(method=transformation)

    faces = umikryFaceDetector.detect(image)

    if familar_faces:
        umikryFaceRecognizer = UmikryFaceRecognizer()
        faces = umikryFaceRecognizer.recognize_faces(image, faces, familar_faces)

    image = umikryFaceTransformator.transform(image, faces)

    return image
