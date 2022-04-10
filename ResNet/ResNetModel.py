import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
import cv2

from Utils import IsModelExist


class ResNetModel:

    def __init__(self):
        self.dataset = None
        self.resnet_model = None
        self.load_resnet_model()

    def load_resnet_model(self):
        self.dataset = tf.keras.utils.image_dataset_from_directory(
            'ImgFlip575k_Dataset-master/dataset/img',
            labels='inferred',
            label_mode='categorical',
        )

        if IsModelExist(model_name='resnet_model'):
            self.resnet_model = load_model('resnet_model')
        else:

            self.resnet_model = Sequential()

            pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                              input_shape=(256, 256, 3),
                                                              pooling='avg', classes=5,
                                                              weights='imagenet')
            for layer in pretrained_model.layers:
                layer.trainable = False

            self.resnet_model.add(pretrained_model)

            self.resnet_model.add(Flatten())
            self.resnet_model.add(Dense(512, activation='relu'))
            self.resnet_model.add(Dense(100, activation='softmax'))

            print(self.resnet_model.summary())

            self.resnet_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',
                                      metrics=['accuracy'])

            self.resnet_model.fit(self.dataset, epochs=10)

            self.resnet_model = tf.keras.models.Sequential(self.resnet_model.layers[:-2])

            self.resnet_model.save('resnet_model')

    def predict(self, image_path):

        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, (256, 256))
        image = np.expand_dims(image_resized, axis=0)

        prediction = self.resnet_model.predict(image)
        print(prediction)

        return prediction
