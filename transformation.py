# from generator import build_generator
import cv2


def transform(image, method='blur'):
    if method == 'blur':
        return cv2.blur(image, (25, 25))
    elif method == 'generator':
        raise NotImplementedError
