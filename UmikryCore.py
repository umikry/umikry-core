'''
Copyright (c) 2018, umikry.com
License AGPL-3.0

This code is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License, version 3,
as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License, version 3,
along with this program.  If not, see <http://www.gnu.org/licenses/>
'''

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
