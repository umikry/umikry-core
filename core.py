from detection import UmikryFaceDetector
from transformation import transform


def umikry(image, detection='caffe', transformation='blur'):
    umikryFaceDetector = UmikryFaceDetector(method=detection)
    faces = umikryFaceDetector.detect(image)

    for _, (x, y, w, h) in faces.items():
        if h < image.shape[0] and w < image.shape[1]:
            image[y:h, x:w] = transform(image[y:h, x:w], method=transformation)

    return image
