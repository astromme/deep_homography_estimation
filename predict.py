from keras.models import load_model
from keras import backend as K
from generate_dataset import data_generator, data_generator_similarity, batch_size, num_data_samples
import itertools
from glob import glob
import sys
from matplotlib import pyplot as plt
import cv2
import numpy as np

file1, file2 = sys.argv[1:]

def load_image(filename):
    image = plt.imread(filename)
    image = cv2.resize(image, (128, 128))
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return image

image1 = load_image(file1)
image2 = load_image(file2)


#model = load_model('model-binary-similar-image.h5')
model = load_model('model.h5')

predictions = model.predict(np.expand_dims(np.dstack((image1, image2)), axis=0), batch_size=1).reshape((-1,))
print(predictions)
