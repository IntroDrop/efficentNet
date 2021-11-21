# 图像编码和格式预处理
import math

import numpy as np

from cfg import img_size, bool_random_flip_left_right, bool_random_flip_up_down, bool_random_brightness, \
    bool_random_contrast, bool_random_hue, bool_random_saturation, bool_random_crop, cutmix_rate, BATCH_SIZE, \
    nb_classes, mixup_rate, gridmask_rate


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)

    if label is None:
        return image
    else:
        return image, label


# 图像数据增强

# 只能写入test data 也能用的 aug
def data_aug(image, label=None, image_size=(img_size, img_size)):
    if bool_random_flip_left_right:
        image = tf.image.random_flip_left_right(image)
    if bool_random_flip_up_down:
        image = tf.image.random_flip_up_down(image)
    if bool_random_brightness:
        image = tf.image.random_brightness(image, 0.2)
    if bool_random_contrast:
        image = tf.image.random_contrast(image, 0.6, 1.4)
    if bool_random_hue:
        image = tf.image.random_hue(image, 0.07)
    if bool_random_saturation:
        image = tf.image.random_saturation(image, 0.5, 1.5)
    if bool_random_crop:
        image = tf.image.resize(image, [int(1.2 * ele) for ele in image_size])
        image = tf.image.random_crop(image, image_size + [3])

    if label is None:
        return image
    else:
        return image, label


import tensorflow as tf, tensorflow.keras.backend as K


def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies

    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.

    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    rotation_matrix = tf.reshape(tf.concat([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0), [3, 3])

    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape(tf.concat([one, s2, zero, zero, c2, zero, zero, zero, one], axis=0), [3, 3])

    # ZOOM MATRIX
    zoom_matrix = tf.reshape(
        tf.concat([one / height_zoom, zero, zero, zero, one / width_zoom, zero, zero, zero, one], axis=0), [3, 3])

    # SHIFT MATRIX
    shift_matrix = tf.reshape(tf.concat([one, zero, height_shift, zero, one, width_shift, zero, zero, one], axis=0),
                              [3, 3])

    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))


def rotation_transform(image, label):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    DIM = img_size
    XDIM = DIM % 2  # fix for size 331

    rot = 15. * tf.random.normal([1], dtype='float32')
    shr = 5. * tf.random.normal([1], dtype='float32')
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / 10.
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / 10.
    h_shift = 16. * tf.random.normal([1], dtype='float32')
    w_shift = 16. * tf.random.normal([1], dtype='float32')

    # GET TRANSFORMATION MATRIX
    m = get_mat(rot, shr, h_zoom, w_zoom, h_shift, w_shift)

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype='int32')
    idx = tf.stack([x, y, z])

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [DIM, DIM, 3]), label


# 在batch内部互相随机取图
def cutmix(image, label, PROBABILITY=cutmix_rate):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with cutmix applied
    label = tf.cast(label, tf.float32)

    DIM = img_size
    imgs = []
    labs = []

    for j in range(BATCH_SIZE):
        # random_uniform( shape, minval=0, maxval=None)
        # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)

        # CHOOSE RANDOM IMAGE TO CUTMIX WITH
        k = tf.cast(tf.random.uniform([], 0, BATCH_SIZE), tf.int32)

        # CHOOSE RANDOM LOCATION
        # 选一个随机的中心点
        x = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)

        # Beta(1, 1)等于均匀分布
        b = tf.random.uniform([], 0, 1)  # this is beta dist with alpha=1.0

        # P只随机出0或1，就是裁剪或是不裁剪
        WIDTH = tf.cast(DIM * tf.math.sqrt(1 - b), tf.int32) * P
        ya = tf.math.maximum(0, y - WIDTH // 2)
        yb = tf.math.minimum(DIM, y + WIDTH // 2)
        xa = tf.math.maximum(0, x - WIDTH // 2)
        xb = tf.math.minimum(DIM, x + WIDTH // 2)

        # MAKE CUTMIX IMAGE
        one = image[j, ya:yb, 0:xa, :]
        two = image[k, ya:yb, xa:xb, :]
        three = image[j, ya:yb, xb:DIM, :]
        # 得出了ya:yb区间内的输出图像
        middle = tf.concat([one, two, three], axis=1)
        # 得到了完整输出图像
        img = tf.concat([image[j, 0:ya, :, :], middle, image[j, yb:DIM, :, :]], axis=0)
        imgs.append(img)

        # MAKE CUTMIX LABEL
        # 按面积来加权的
        a = tf.cast(WIDTH * WIDTH / DIM / DIM, tf.float32)
        lab1 = label[j,]
        lab2 = label[k,]
        labs.append((1 - a) * lab1 + a * lab2)

    image2 = tf.reshape(tf.stack(imgs), (BATCH_SIZE, DIM, DIM, 3))
    label2 = tf.reshape(tf.stack(labs), (BATCH_SIZE, nb_classes))
    return image2, label2


def mixup(image, label, PROBABILITY=mixup_rate):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with mixup applied
    DIM = img_size

    imgs = []
    labs = []
    for j in range(BATCH_SIZE):

        # CHOOSE RANDOM
        k = tf.cast(tf.random.uniform([], 0, BATCH_SIZE), tf.int32)
        a = tf.random.uniform([], 0, 1)  # this is beta dist with alpha=1.0

        # 根据概率抽取执不执行mixup
        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)
        if P == 1:
            a = 0.

        # MAKE MIXUP IMAGE
        img1 = image[j,]
        img2 = image[k,]
        imgs.append((1 - a) * img1 + a * img2)

        # MAKE CUTMIX LABEL
        lab1 = label[j,]
        lab2 = label[k,]
        labs.append((1 - a) * lab1 + a * lab2)

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs), (BATCH_SIZE, DIM, DIM, 3))
    label2 = tf.reshape(tf.stack(labs), (BATCH_SIZE, nb_classes))
    return image2, label2


# gridmask
def transform(image, inv_mat, image_shape):
    h, w, c = image_shape
    cx, cy = w // 2, h // 2
    new_xs = tf.repeat(tf.range(-cx, cx, 1), h)
    new_ys = tf.tile(tf.range(-cy, cy, 1), [w])
    new_zs = tf.ones([h * w], dtype=tf.int32)
    old_coords = tf.matmul(inv_mat, tf.cast(tf.stack([new_xs, new_ys, new_zs]), tf.float32))
    old_coords_x, old_coords_y = tf.round(old_coords[0, :] + w // 2), tf.round(old_coords[1, :] + h // 2)
    clip_mask_x = tf.logical_or(old_coords_x < 0, old_coords_x > w - 1)
    clip_mask_y = tf.logical_or(old_coords_y < 0, old_coords_y > h - 1)
    clip_mask = tf.logical_or(clip_mask_x, clip_mask_y)
    old_coords_x = tf.boolean_mask(old_coords_x, tf.logical_not(clip_mask))
    old_coords_y = tf.boolean_mask(old_coords_y, tf.logical_not(clip_mask))
    new_coords_x = tf.boolean_mask(new_xs + cx, tf.logical_not(clip_mask))
    new_coords_y = tf.boolean_mask(new_ys + cy, tf.logical_not(clip_mask))
    old_coords = tf.cast(tf.stack([old_coords_y, old_coords_x]), tf.int32)
    new_coords = tf.cast(tf.stack([new_coords_y, new_coords_x]), tf.int64)
    rotated_image_values = tf.gather_nd(image, tf.transpose(old_coords))
    rotated_image_channel = list()
    for i in range(c):
        vals = rotated_image_values[:, i]
        sparse_channel = tf.SparseTensor(tf.transpose(new_coords), vals, [h, w])
        rotated_image_channel.append(tf.sparse.to_dense(sparse_channel, default_value=0, validate_indices=False))
    return tf.transpose(tf.stack(rotated_image_channel), [1, 2, 0])


def random_rotate(image, angle, image_shape):
    def get_rotation_mat_inv(angle):
        # transform to radian
        angle = math.pi * angle / 180
        cos_val = tf.math.cos(angle)
        sin_val = tf.math.sin(angle)
        one = tf.constant([1], tf.float32)
        zero = tf.constant([0], tf.float32)
        rot_mat_inv = tf.concat([cos_val, sin_val, zero, -sin_val, cos_val, zero, zero, zero, one], axis=0)
        rot_mat_inv = tf.reshape(rot_mat_inv, [3, 3])
        return rot_mat_inv

    angle = float(angle) * tf.random.normal([1], dtype='float32')
    rot_mat_inv = get_rotation_mat_inv(angle)
    return transform(image, rot_mat_inv, image_shape)


def GridMask(image_height, image_width, d1, d2, rotate_angle=1, ratio=0.5):
    h, w = image_height, image_width
    hh = int(np.ceil(np.sqrt(h * h + w * w)))
    hh = hh + 1 if hh % 2 == 1 else hh
    d = tf.random.uniform(shape=[], minval=d1, maxval=d2, dtype=tf.int32)
    l = tf.cast(tf.cast(d, tf.float32) * ratio + 0.5, tf.int32)

    st_h = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)
    st_w = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)

    y_ranges = tf.range(-1 * d + st_h, -1 * d + st_h + l)
    x_ranges = tf.range(-1 * d + st_w, -1 * d + st_w + l)

    for i in range(0, hh // d + 1):
        s1 = i * d + st_h
        s2 = i * d + st_w
        y_ranges = tf.concat([y_ranges, tf.range(s1, s1 + l)], axis=0)
        x_ranges = tf.concat([x_ranges, tf.range(s2, s2 + l)], axis=0)

    x_clip_mask = tf.logical_or(x_ranges < 0, x_ranges > hh - 1)
    y_clip_mask = tf.logical_or(y_ranges < 0, y_ranges > hh - 1)
    clip_mask = tf.logical_or(x_clip_mask, y_clip_mask)

    x_ranges = tf.boolean_mask(x_ranges, tf.logical_not(clip_mask))
    y_ranges = tf.boolean_mask(y_ranges, tf.logical_not(clip_mask))

    hh_ranges = tf.tile(tf.range(0, hh), [tf.cast(tf.reduce_sum(tf.ones_like(x_ranges)), tf.int32)])
    x_ranges = tf.repeat(x_ranges, hh)
    y_ranges = tf.repeat(y_ranges, hh)

    y_hh_indices = tf.transpose(tf.stack([y_ranges, hh_ranges]))
    x_hh_indices = tf.transpose(tf.stack([hh_ranges, x_ranges]))

    y_mask_sparse = tf.SparseTensor(tf.cast(y_hh_indices, tf.int64), tf.zeros_like(y_ranges), [hh, hh])
    y_mask = tf.sparse.to_dense(y_mask_sparse, 1, False)

    x_mask_sparse = tf.SparseTensor(tf.cast(x_hh_indices, tf.int64), tf.zeros_like(x_ranges), [hh, hh])
    x_mask = tf.sparse.to_dense(x_mask_sparse, 1, False)

    mask = tf.expand_dims(tf.clip_by_value(x_mask + y_mask, 0, 1), axis=-1)

    mask = random_rotate(mask, rotate_angle, [hh, hh, 1])
    mask = tf.image.crop_to_bounding_box(mask, (hh - h) // 2, (hh - w) // 2, image_height, image_width)

    return mask


def apply_grid_mask(image, image_shape, PROBABILITY=gridmask_rate):
    AugParams = {
        'd1': 100,
        'd2': 160,
        'rotate': 45,
        'ratio': 0.3
    }

    mask = GridMask(image_shape[0], image_shape[1], AugParams['d1'], AugParams['d2'], AugParams['rotate'],
                    AugParams['ratio'])
    if image_shape[-1] == 3:
        mask = tf.concat([mask, mask, mask], axis=-1)
        mask = tf.cast(mask, tf.float32)
        # print(mask.shape) # (299,299,3)

    # 整个batch启用或者不启用
    P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)
    if P == 1:
        return image * mask
    else:
        return image


def gridmask(img_batch, label_batch):
    return apply_grid_mask(img_batch, (img_size, img_size, 3)), label_batch
