from keras.models import load_model
from keras import backend as K
from generate_dataset import data_generator, data_generator_similarity, batch_size, num_data_samples
import itertools
from glob import glob
import sys
from matplotlib import pyplot as plt
import cv2
import numpy as np

model = load_model('model.h5')
#model = load_model('model-binary-similar-image.h5')

def preprocess_image(image):
    image = cv2.resize(image, (128, 128))

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return image

def predict(image, compare_image):
    image = preprocess_image(image)
    compare_image = preprocess_image(compare_image)

    predictions = model.predict(np.expand_dims(np.dstack((image, compare_image)), axis=0), batch_size=1).reshape((-1,))

    return predictions

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    compare_image = None
    while True:
        ret_val, img = cam.read()
        if not compare_image:
            compare_image = img
        if mirror:
            img = cv2.flip(img, 1)
        similarity = predict(img, compare_image)
        cv2.imshow('similarity: {}'.format(similarity), img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()

def main():
	show_webcam(mirror=True)

if __name__ == '__main__':
    main()
