from keras.models import load_model
from keras import backend as K
from generate_dataset import data_generator, data_generator_similarity, batch_size, num_data_samples
import itertools
from glob import glob


# model = load_model('model-binary-similar-image.h5')
model = load_model('model.h5')

data_generator = data_generator_similarity(glob("/home/astromme/Code/city-vision/FastMaskRCNN/data/coco/val2014/*.jpg")[:num_data_samples])

images, labels = next(data_generator)


predictions = model.predict(images, batch_size=batch_size).reshape((-1,))

accurate = 0
wrong = 0
for prediction, label in zip(predictions, labels):
    if prediction < 0.5 and label == 0 or prediction >= 0.5 and label == 1:
        print('correct')
        accurate += 1
    else:
        print('incorrect')
        wrong += 1

print('accuracy: {}'.format(accurate/(accurate+wrong)))

# print(predictions)
# print(labels)

# tf_session = K.get_session()
# print(euclidean_distance(predictions, datum[1]).eval(session=tf_session))
