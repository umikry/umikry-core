import cv2
import dlib
import numpy as np
import math
import os
from UmikryDCGAN import build_generator
from UmikryFaceRecognizer import UmikryFaceRecognizer
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D


def next_multiply_of_n(x, n):
    return math.ceil(x / n) * n


def pad_to_next_multiply_of_n(image, n):
    diff_y = next_multiply_of_n(image.shape[0], n) - image.shape[0]
    diff_x = next_multiply_of_n(image.shape[1], n) - image.shape[1]

    border_top = int(diff_y / 2)
    border_bottom = diff_y - border_top
    border_left = int(diff_x / 2)
    border_right = diff_x - border_left

    padded_image = cv2.copyMakeBorder(image, border_top, border_bottom, border_left, border_right,
                                      cv2.BORDER_REFLECT)
    return padded_image, (border_top, border_bottom, border_left, border_right)


def shape_to_np(shape):
    coords = np.zeros((68, 2), dtype=int)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    coords = coords.tolist()
    return [tuple(coord) for coord in coords]


def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True


def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return dst


def warp_triangle(original_image, image, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1_rect = original_image[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])

    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)

    img2_rect = img2_rect * mask

    # Copy triangular region of the rectangular patch to the output image
    gimage = image.copy()
    gimage[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = gimage[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - mask)

    gimage[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = gimage[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect

    return gimage


class UmikryFaceTransformator(object):
    def __init__(self, method='AUTOENCODER'):
        self.shape_predictor = UmikryFaceRecognizer(shape_predictor='BIG').shape_predictor
        if method == 'AUTOENCODER':
            self.method = method
            self.model = self.build_autoencoder()
            self.model.load_weights(os.path.join('models', 'autoencoder_weights.h5'), by_name=True)
        elif method == 'GAN':
            self.method = method
            generator_input, generator_output = build_generator()
            self.model = Model(generator_input, generator_output)
            self.model.load_weights(os.path.join('models', 'generator_weights.h5'))
        elif method == 'BLUR':
            self.method = method
        else:
            raise ValueError('{} is an invalid method use \'AUTOENCODER\', \'GAN\' or \'BLUR\' instead'.format(self.method))

    def build_autoencoder(self):
        input_image = Input(shape=(None, None, 3), name='e_input')
        x = Conv2D(32, 3, padding='same', name='e_conv1', activation='relu')(input_image)
        x = MaxPooling2D(padding='same', name='e_pool1')(x)
        x = Conv2D(64, 3, padding='same', name='e_conv2', activation='relu')(x)
        x = MaxPooling2D(padding='same', name='e_pool2')(x)
        x = Conv2D(128, 3, padding='same', name='e_conv3', activation='relu')(x)
        x = MaxPooling2D(padding='same', name='e_pool3')(x)
        encoder = Conv2D(128, 3, padding='same', name='encoded')(x)

        x = Conv2D(128, 3, padding='same', name='d_conv1', activation='relu')(encoder)
        x = Conv2DTranspose(128, 3, padding='same', strides=2, name='d_up1', activation='relu')(x)
        x = Conv2D(64, 3, padding='same', name='d_conv2', activation='relu')(x)
        x = Conv2DTranspose(64, 3, padding='same', strides=2, name='d_up2', activation='relu')(x)
        x = Conv2D(32, 3, padding='same', name='d_conv3', activation='relu')(x)
        x = Conv2DTranspose(32, 3, padding='same', strides=2, name='d_up3', activation='relu')(x)
        x = Conv2DTranspose(32, 3, padding='same', strides=1, name='d_denoise', activation='relu')(x)
        x = Conv2D(3, 5, padding='same', name='d_pre_decoded', activation='sigmoid')(x)
        decoder = Conv2D(3, 1, padding='same', name='decoded', activation='sigmoid')(x)

        return Model(inputs=input_image, outputs=decoder)

    def transform(self, image, faces):
        orig_image = image.copy()
        if self.method == 'BLUR':
            for _, (x, y, w, h) in faces.items():
                if h < image.shape[0] and w < image.shape[1]:
                    image[y:h, x:w] = cv2.blur(image[y:h, x:w], (25, 25))

            return image
        elif self.method == 'AUTOENCODER':
            prediction_list = []
            image_list = []
            for _, (x, y, w, h) in faces.items():
                if h < image.shape[0] and w < image.shape[1]:
                    blurry = cv2.blur(image[y:h, x:w], (25, 25)) / 255.0
                    blurry, border = pad_to_next_multiply_of_n(blurry, 8)

                    prediction = self.model.predict(np.array([blurry]))[0]
                    prediction = prediction[border[0]:(prediction.shape[0] - border[1]),
                                            border[2]:(prediction.shape[1] - border[3]), :]

                    image[y:h, x:w] = (prediction * 255).astype(np.uint8)

            return image, prediction_list, image_list
        elif self.method == 'GAN':
            prediction_list = []
            image_list = []
            rect_list = []
            # left, top, right, bottom
            for _, (x, y, w, h) in faces.items():
                if h < image.shape[0] and w < image.shape[1]:
                    section, border = pad_to_next_multiply_of_n(image[y:h, x:w], 4)
                    section = cv2.resize(section, (section.shape[1] // 4, section.shape[0] // 4)) / 255.0

                    prediction = self.model.predict(np.array([section]))[0] * 0.5 + 0.5

                    prediction = (prediction[border[0]:(prediction.shape[0] - border[1]),
                                             border[2]:(prediction.shape[1] - border[3]), :] * 255)

                    prediction = prediction.astype(np.uint8)
                    prediction_list.append(prediction)
                    image_list.append(image[y:h, x:w])
                    face = self.__smoth_replace(image[y:h, x:w], prediction)
                    image[y:h, x:w] = face

                    rect_list.append(dlib.rectangle(x, y, w, h))
            output = self.__swap_image(orig_image, image, rect_list)
            return output

    def __swap_image(self, orig_image, image, rect_list):
        image_warped = image.copy()
        shape_dict = {}
        for a, rect in enumerate(rect_list):
            print(a)
            shape_dict["original"] = shape_to_np(self.shape_predictor(orig_image, rect))
            shape_dict["gan"] = shape_to_np(self.shape_predictor(image, rect))
            hull_index_1 = []
            hull_index_2 = []
            hull_index = cv2.convexHull(np.array(shape_dict["original"]), returnPoints=False)
            for i in range(0, len(hull_index)):
                hull_index_1.append(shape_dict["original"][int(hull_index[i])])
                hull_index_2.append(shape_dict["gan"][int(hull_index[i])])
            rect_gan = (0, 0, orig_image.shape[1], orig_image.shape[0])
            subdiv = cv2.Subdiv2D(rect_gan)
            for p in hull_index_2:
                subdiv.insert(p)
            triangle_list = subdiv.getTriangleList()
            delaunay_tri = []
            pt = []
            for t in triangle_list:
                pt.append((t[0], t[1]))
                pt.append((t[2], t[3]))
                pt.append((t[4], t[5]))

                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])
                if rect_contains(rect_gan, pt1) and rect_contains(rect_gan, pt2) and rect_contains(rect_gan, pt3):
                    ind = []
                    # Get face-points (from 68 face detector) by coordinates
                    for j in range(0, 3):
                        for k in range(0, len(hull_index_2)):
                            if abs(pt[j][0] - hull_index_2[k][0]) < 1.0 and abs(pt[j][1] - hull_index_2[k][1]) < 1.0:
                                ind.append(k)
                                # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph
                    if len(ind) == 3:
                        delaunay_tri.append((ind[0], ind[1], ind[2]))

                    pt = []
            for i in range(0, len(delaunay_tri)):
                t1 = []
                t2 = []

                # get points for img1, img2 corresponding to the triangles
                for j in range(0, 3):
                    t1.append(hull_index_1[delaunay_tri[i][j]])
                    t2.append(hull_index_2[delaunay_tri[i][j]])
                image_warped = warp_triangle(image, image_warped, t1, t2)

            hull8_u = []
            for i in range(0, len(hull_index_2)):
                hull8_u.append((hull_index_2[i][0], hull_index_2[i][1]))
            mask = np.zeros(image.shape, dtype=image.dtype)
            cv2.fillConvexPoly(mask, np.int32(hull8_u), (255, 255, 255))
            r = cv2.boundingRect(np.float32([hull_index_2]))
            center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))
            output = cv2.seamlessClone(np.uint8(image_warped), orig_image, mask, center, cv2.NORMAL_CLONE)
            image[rect.top():rect.bottom(), rect.left():rect.right()] = output[rect.top():rect.bottom(),
                                                                               rect.left():rect.right()]

        return image

    def __smoth_replace(self, image, face):
        mean_values = np.mean(np.mean(image, axis=0), axis=0)
        mean_shift = mean_values - np.mean(np.mean(face, axis=0), axis=0)

        for i in range(face.shape[2]):
            face[:, :, i] = face[:, :, i] + mean_shift[i]
            face[:, :, i] = np.where(face[:, :, i] > 0, face[:, :, i], 0)
            face[:, :, i] = np.where(face[:, :, i] < 255, face[:, :, i], 255)

        steps = [(0, 0.2), (2, 0.4), (4, 0.5), (8, 0.6), (10, 1.0)]
        smoth_face = np.ones_like(face)
        for step, intensity in steps:
            if step == 0:
                background = np.uint8(image * (1 - intensity))
                foreground = np.uint8(face * intensity)
                smoth_face = background + foreground
            else:
                background = np.uint8(image[step:-step, step:-step] * (1 - intensity))
                foreground = np.uint8(face[step:-step, step:-step] * intensity)
                smoth_face[step:-step, step:-step] = background + foreground

        return smoth_face
