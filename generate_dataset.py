from glob import glob
from matplotlib import pyplot as plt
import cv2
import random
import numpy as np
from numpy.linalg import inv
import tqdm
import keras

num_data_samples = 100000

loc_list = glob("./ms_coco_test_images/*.jpg")
loc_list = glob("/home/astromme/Code/city-vision/FastMaskRCNN/data/coco/train2014/*.jpg")[:num_data_samples]

batch_size = 128

images = {}
# images = []
# for loc in tqdm.tqdm(loc_list):
#     image = plt.imread(loc)
#     image = cv2.resize(image, (320, 240))
#     if len(image.shape) == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     images.append(image)


def get_image(index):
    try:
        return images[loc_list[index]]
    except KeyError:
        image = plt.imread(loc_list[index])
        image = cv2.resize(image, (320, 240))
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        images[loc_list[index]] = image

        return image


def data_generator():
    while True:
        batch_images = [random.choice(images) for i in range(batch_size)]
        datums = [generate_datum(image) for image in batch_images]

        inputs, targets = list(zip(*datums))

        inputs, targets = np.asarray(inputs), np.asarray(targets)
        targets = targets.reshape((-1, 8))

        yield inputs, targets

def data_generator_similarity(loc_list):
    while True:
        inputs, targets = [], []
        for i in range(batch_size):
            index1 = random.randint(0, len(loc_list)-1)
            if random.randint(0, 1) == 1:
                index2 = index1
            else:
                index2 = random.randint(0, len(loc_list)-1)

            image1 = np.copy(get_image(index1))
            image2 = np.copy(get_image(index2))

            image1 = modify_brightness_gray(image1, random.randint(-50, 50))
            image2 = modify_brightness_gray(image2, random.randint(-50, 50))

            assert(np.max(image1) <= 255)
            assert(np.min(image1) >= 0)

            distorted1 = np.dsplit(generate_datum(image1)[0], 2)[1]
            distorted2 = np.dsplit(generate_datum(image2)[0], 2)[1]

            if index1 == index2:
                output = 1
            else:
                output = 0

            inputs.append(np.dstack((distorted1, distorted2)))
            targets.append(output)


        # targets = keras.utils.to_categorical(targets, num_classes=2)
        inputs, targets = np.asarray(inputs), np.asarray(targets)

        yield inputs, targets

def modify_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if value > 0:
        value = np.uint8(value)
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        value = np.uint8(abs(value))
        lim = 0 + value
        v[v < lim] = 0
        v[v >= lim] -= value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def modify_brightness_gray(img, value=30):
    if value > 0:
        value = np.uint8(value)
        lim = 255 - value
        img[img > lim] = 255
        img[img <= lim] += value
    else:
        value = np.uint8(abs(value))
        lim = 0 + value
        img[img < lim] = 0
        img[img >= lim] -= value

    return img

def generate_datum(image):
    rho          = 32
    patch_size   = 128

    top_point    = (32,32)
    left_point   = (patch_size+32, 32)
    bottom_point = (patch_size+32, patch_size+32)
    right_point  = (32, patch_size+32)

    annotated_image = image.copy()

    four_points = [top_point, left_point, bottom_point, right_point]

    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append((point[0] + random.randint(-rho,rho), point[1]+random.randint(-rho,rho)))


    # h is the homography matrix
    H = cv2.getPerspectiveTransform( np.float32(four_points), np.float32(perturbed_four_points) )
    H_inverse = inv(H)

    warped_image = cv2.warpPerspective(image,H_inverse, (320,240))

    Ip1 = annotated_image[top_point[1]:bottom_point[1],top_point[0]:bottom_point[0]]
    Ip2 = warped_image[top_point[1]:bottom_point[1],top_point[0]:bottom_point[0]]

    training_image = np.dstack((Ip1, Ip2))
    H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))

    datum = (training_image, H_four_points)

    return datum
