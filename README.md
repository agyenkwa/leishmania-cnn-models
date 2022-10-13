# leishmania-cnn-models
Deep learning CNN models (ResNet50, VGG16 and InceptionV3) trained on parasite images to aid differentiation between leishmania and other parasites.

Each zipped folder contains the trained models

1. Download the zipped folders 
2. Unzip into a directory
3. load them using a script in that same directory:

from tensorflow import keras

model = keras.models.load_model('Unzipped folder name')

model.summary
