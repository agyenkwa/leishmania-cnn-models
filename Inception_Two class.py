#!/usr/bin/ml python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 12:46:11 2021

@author: Kwasi Agyenkwa-Mawuli
"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import roc_auc_score, cohen_kappa_score, matthews_corrcoef



#define size to which images are to be resized
IMAGE_SIZE = [224, 224] # feel free to change depending on dataset

# training config:
epochs = 5
batch_size = 32

#define paths
pos_path = '/home/waccbip/Desktop/HPCleish/Positives_all'
neg_path = '/home/waccbip/Desktop/HPCleish/Negatives_main'

# Use glob to grab images from path .jpg or jpeg
pos_files = glob(pos_path + '/*')
neg_files = glob(neg_path + '/*')

# Preparing Labels
pos_labels = []
neg_labels = []

pos_images=[]
neg_images=[]



for i in range(len(pos_files)):
  image = cv2.imread(pos_files[i])
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image,(224,224))
  pos_images.append(image)
  pos_labels.append('Leish')
  
for i in range(len(neg_files)):
  image = cv2.imread(neg_files[i])
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image,(224,224))
  neg_images.append(image)
  neg_labels.append('Others')

# look at a random image for fun
def plot_images(images, title):
    nrows, ncols = 5, 8
    figsize = [10, 6]

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, facecolor=(1, 1, 1))

    for i, axi in enumerate(ax.flat):
        axi.imshow(images[i])
        axi.set_axis_off()

    plt.suptitle(title, fontsize=24)
    plt.tight_layout(pad=0.2, rect=[0, 0, 1, 0.9])
    plt.show()
plot_images(pos_images, 'Leishmania parasite present')
plot_images(neg_images, 'Leishmania parasite absent')

# normalize to interval of [0,1]
pos_images = np.array(pos_images) / 255
neg_images = np.array(neg_images) / 255

# split into training and testing
pos_x_train, pos_x_test, pos_y_train, pos_y_test = train_test_split(
    pos_images, pos_labels, test_size=0.2)
neg_x_train, neg_x_test, neg_y_train, neg_y_test = train_test_split(
    neg_images, neg_labels, test_size=0.2)


X_train = np.concatenate((neg_x_train, pos_x_train), axis=0)
X_test = np.concatenate((neg_x_test, pos_x_test), axis=0)
y_train = np.concatenate((neg_y_train, pos_y_train), axis=0)
y_test = np.concatenate((neg_y_test, pos_y_test), axis=0)

# make labels into categories - either 0 or 1
y_train = LabelBinarizer().fit_transform(y_train)
y_train = to_categorical(y_train)

y_test = LabelBinarizer().fit_transform(y_test)
y_test = to_categorical(y_test)

plot_images(pos_x_train, 'X_train')
plot_images(neg_x_test, 'X_test')
# y_train and y_test contain class lables 0 and 1 representing Leish and Others for X_train and X_test

inception = InceptionV3(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

outputs = inception.output
outputs = Flatten(name="flatten")(outputs)
outputs = Dropout(0.5)(outputs)
outputs = Dense(2, activation="softmax")(outputs)

model = Model(inputs=inception.input, outputs=outputs)

for layer in inception.layers:
    layer.trainable = False

model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
)

train_aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

model.summary()

history = model.fit(train_aug.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    validation_steps=len(X_test) / 32,
                    steps_per_epoch=len(X_train) / 32,
                    epochs=5)

model.save('Inception_Two class')
model = load_model('Inception_Two class')

y_pred = model.predict(X_test, batch_size=batch_size)

prediction=y_pred[0:5]
for index, probability in enumerate(prediction):
  if probability[1] > 0.5:
        plt.title('%.2f' % (probability[1]*100) + '% Leish')
  else:
        plt.title('%.2f' % ((1-probability[1])*100) + '% Others')
  plt.imshow(X_test[index])
  plt.show()
  
  
# Convert to Binary classes
y_pred_bin = np.argmax(y_pred, axis=1)
y_test_bin = np.argmax(y_test, axis=1)

fpr, tpr, thresholds = roc_curve(y_test_bin, y_pred_bin)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for InceptionV3')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)

def plot_confusion_matrix(normalize):
  classes = ['Leish','Others']
  tick_marks = [0.5,1.5]
  cn = confusion_matrix(y_test_bin, y_pred_bin,normalize=normalize)
  sns.heatmap(cn,cmap='plasma',annot=True)
  plt.xticks(tick_marks, classes)
  plt.yticks(tick_marks, classes)
  plt.title('Confusion Matrix')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

print('Confusion Matrix without Normalization')
plot_confusion_matrix(normalize=None)

print('Confusion Matrix with Normalized Values')
plot_confusion_matrix(normalize='true')

from sklearn.metrics import classification_report
report = (classification_report(y_test_bin, y_pred_bin))
print(report)
r = open('report_inception.txt', 'w')
r.write(report)
r.close()

MCC = matthews_corrcoef(y_test_bin, y_pred_bin)
ROC_AUC = roc_auc_score(y_test_bin, y_pred_bin)
Kappa = cohen_kappa_score(y_test_bin, y_pred_bin)


print('MCC: %.3f' % MCC)
print('\n')

print('COHEN KAPPA: %.3f' % Kappa)
print('\n')

print('ROC AUC SCORE: %.3f' % ROC_AUC)
print('\n')

a = MCC, ROC_AUC, Kappa
np.savetxt("parameters_inception.txt", a)

plt.figure(figsize=(10,10))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('inception_accuracy.png')
plt.show()

plt.figure(figsize=(10,10))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('inception_loss.png')
plt.show()
