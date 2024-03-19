# -*- coding: utf-8 -*-
"""Brain Tumor.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1j2FIupxPTMq7kSuJSKrkA3DgXtxToP14
"""

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image, ImageEnhance

# For ML Models
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.models import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.preprocessing.image import load_img

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model

from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from IPython.core.display import Image

# For Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Miscellaneous
from tqdm import tqdm
import random
import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import os

os.environ["KAGGLE_USERNAME"] = "shahinsaifi"
os.environ["KAGGLE_KEY"] = "b59d337c916648f0b2e477242ae3dc4c"

!kaggle datasets download masoudnickparvar/brain-tumor-mri-dataset

!unzip brain-tumor-mri-dataset

train_dir = '/content/Training/'
test_dir = '/content/Testing/'

train_paths = []
train_labels = []

for label in os.listdir(train_dir):
    for image in os.listdir(train_dir+label):
        train_paths.append(train_dir+label+'/'+image)
        train_labels.append(label)

train_paths, train_labels = shuffle(train_paths, train_labels)

plt.figure(figsize=(10, 6), facecolor='grey')
colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853']
plt.rcParams.update({'font.size': 12})
patches, texts, autotexts = plt.pie([len([x for x in train_labels if x=='pituitary']),
                                    len([x for x in train_labels if x=='notumor']),
                                    len([x for x in train_labels if x=='meningioma']),
                                    len([x for x in train_labels if x=='glioma'])],
                                   labels=None,  # Remove the labels
                                   colors=colors, autopct='%.1f%%', explode=(0.025, 0.025, 0.025, 0.025),
                                   startangle=30)

plt.title('Distribution of Training Data', fontdict={'size': 16, 'color': 'white', 'weight': 'bold'})
plt.legend(patches, ['pituitary', 'notumor', 'meningioma', 'glioma'], loc='upper left', bbox_to_anchor=(1, 1), facecolor='grey')
plt.show()

test_paths = []
test_labels = []

for label in os.listdir(test_dir):
    for image in os.listdir(test_dir+label):
        test_paths.append(test_dir+label+'/'+image)
        test_labels.append(label)

test_paths, test_labels = shuffle(test_paths, test_labels)

plt.figure(figsize=(10, 6), facecolor='grey')
colors = ['#fbbc05', '#34a853']
plt.rcParams.update({'font.size': 14, 'text.color': 'black'})
patches, texts, autotexts = plt.pie([len(train_labels), len(test_labels)],
                                   labels=None,  # Remove the labels
                                   colors=colors, autopct='%.1f%%', explode=(0.05, 0),
                                   startangle=30)

plt.title('Distribution of Data', fontdict={'size': 18, 'color': 'black', 'weight': 'bold'})
plt.legend(patches, ['Train', 'Test'], loc='upper left', bbox_to_anchor=(1, 1), facecolor='grey')
plt.show()

def augment_image(image):
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8,1.2))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8,1.2))
    image = np.array(image)/255.0
    return image

IMAGE_SIZE = 128

def open_images(paths):

    images = []
    for path in paths:
        image = load_img(path, target_size=(IMAGE_SIZE,IMAGE_SIZE))
        image = augment_image(image)
        images.append(image)
    return np.array(images)

images = open_images(train_paths[50:59])
labels = train_labels[50:59]
fig = plt.figure(figsize=(12, 6))
for x in range(1, 9):
    fig.add_subplot(2, 4, x)
    plt.axis('off')
    plt.title(labels[x])
    plt.imshow(images[x])
plt.rcParams.update({'font.size': 12})
plt.show()

unique_labels = os.listdir(train_dir)

def encode_label(labels):
    encoded = []
    for x in labels:
        encoded.append(unique_labels.index(x))
    return np.array(encoded)

def decode_label(labels):
    decoded = []
    for x in labels:
        decoded.append(unique_labels[x])
    return np.array(decoded)

def datagen(paths, labels, batch_size=12, epochs=1):
    for _ in range(epochs):
        for x in range(0, len(paths), batch_size):
            batch_paths = paths[x:x+batch_size]
            batch_images = open_images(batch_paths)
            batch_labels = labels[x:x+batch_size]
            batch_labels = encode_label(batch_labels)
            yield batch_images, batch_labels

base_model = VGG16(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3), include_top=False, weights='imagenet')
# Set all layers to non-trainable
for layer in base_model.layers:
    layer.trainable = False
# Set the last vgg block to trainable
base_model.layers[-2].trainable = True
base_model.layers[-3].trainable = True
base_model.layers[-4].trainable = True

model = Sequential()
model.add(Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
model.add(base_model)
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(unique_labels), activation='softmax'))

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001),
             loss='sparse_categorical_crossentropy',
             metrics=['sparse_categorical_accuracy'])

batch_size = 20
steps = int(len(train_paths)/batch_size)
epochs = 10
history = model.fit(datagen(train_paths, train_labels, batch_size=batch_size, epochs=epochs),
                    epochs=epochs, steps_per_epoch=steps)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.grid(True)
plt.plot(history.history['sparse_categorical_accuracy'], '.g-', linewidth=2)
plt.title('Model Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(range(len(history.history['sparse_categorical_accuracy'])))
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.grid(True)
plt.plot(history.history['loss'], '.r-', linewidth=2)
plt.title('Model Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(len(history.history['loss'])))
plt.show()

batch_size = 32
steps = int(len(test_paths)/batch_size)
y_pred = []
y_true = []
for x,y in tqdm(datagen(test_paths, test_labels, batch_size=batch_size, epochs=1), total=steps):
    pred = model.predict(x)
    pred = np.argmax(pred, axis=-1)
    for i in decode_label(pred):
        y_pred.append(i)
    for i in decode_label(y):
        y_true.append(i)

print(classification_report(y_true, y_pred))

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Create a heatmap for visualization
plt.figure(figsize=(5, 3), facecolor='None', frameon=False)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix", fontdict={'size': 14, 'color': 'black'})
plt.xlabel('Predicted Labels', fontdict = {'size':10, 'color': 'black'})
plt.ylabel('True Labels', fontdict = {'size':10, 'color': 'black'})
plt.show()

def names(number):
    if number==0:
        return 'No, Its not a tumour'
    else:
        return 'Its a Tumour'

from PIL import Image

from matplotlib.pyplot import imshow
img = Image.open(r"/content/Training/glioma/Tr-glTr_0002.jpg")
x = np.array(img.resize((128,128)))
x = x.reshape(1,128,128,3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)
print(str(res[0][classification]*100) + '% Conclusion: ' + names(classification))

from matplotlib.pyplot import imshow
img = Image.open(r"/content/Testing/pituitary/Te-piTr_0004.jpg")
x = np.array(img.resize((128,128)))
x = x.reshape(1,128,128,3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)
print(str(res[0][classification]*100) + '% Conclusion: ' + names(classification))

from matplotlib.pyplot import imshow
img = Image.open(r"/content/Training/meningioma/Tr-meTr_0001.jpg")
x = np.array(img.resize((128,128)))
x = x.reshape(1,128,128,3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)
print(str(res[0][classification]*100) + '% Conclusion: ' + names(classification))

def names(number):
    if number==0:
        return 'Its a Tumor'
    else:
        return 'No, Its not a tumor'
from matplotlib.pyplot import imshow
img = Image.open(r"/content/Training/notumor/Tr-noTr_0004.jpg")
x = np.array(img.resize((128,128)))
x = x.reshape(1,128,128,3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)
print(str(res[0][classification]*100) + '% Conclusion: ' + names(classification))

"""#Function to ask user to upload Mri scan image

"""

def predict_tumor_from_upload():
    from google.colab import files

    # Prompt the user to upload an image
    print("Please upload an MRI scan of the brain:")
    uploaded = files.upload()

    # Get the uploaded image file
    image_path = next(iter(uploaded))

    img = Image.open(image_path)
    x = np.array(img.resize((128, 128)))
    x = x.reshape(1, 128, 128, 3)
    res = model.predict_on_batch(x)
    classification = np.where(res == np.amax(res))[1][0]
    imshow(img)

    if classification == 0:
        conclusion = '**Its a Tumor**'
    else:
        conclusion = '**No, Its not a tumor**'

    return str(res[0][classification] * 100) + '% Conclusion: ' + conclusion

result = predict_tumor_from_upload()
print(result)