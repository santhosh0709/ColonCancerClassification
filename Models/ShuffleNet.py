print('---------Shuffle net-----------')

import keras
from keras import layers
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D,Conv2D,MaxPooling2D,BatchNormalization,Concatenate
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, CSVLogger,EarlyStopping,ModelCheckpoint
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import os
import re
from sklearn.utils import shuffle
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import keras.utils as image 

# In[2]:
PATH = "/content/drive/MyDrive/MINOR_PROJECT/DATASET/crchistophenotypes32_32"
print("PWD", PATH)


from keras import backend as K

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    false_positives = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return true_negatives / (true_negatives+false_positives + K.epsilon())

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

data_path = PATH
data_dir_list = sorted_alphanumeric(os.listdir(data_path))
print(data_dir_list)

img_data_list = []

for dataset in sorted_alphanumeric(data_dir_list):
    img_list = sorted_alphanumeric(os.listdir(data_path + '/' + dataset))
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        # print(img)
        img_path = data_path + '/' + dataset + '/' + img
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        #     x = x/255
        # print('Input image shape:', x.shape)
        img_data_list.append(x)

img_data = np.array(img_data_list)
# img_data = img_data.astype('float32')
print(img_data.shape)
img_data = np.rollaxis(img_data, 1, 0)
print(img_data.shape)
img_data = img_data[0]
print(img_data.shape)

# backend
import tensorflow as tf
import tensorflow.keras.backend as k
# from tensorflow import keras.backend as K
# from tensorflow.keras import backend as k

# Don't pre-allocate memory; allocate as-needed
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True

# Create a session with the above options specified.
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# Hyperparameters
batch_size = 64
num_classes = 4
epochs = 500

# Load CIFAR10 Data
num_classes = 4
num_of_samples = img_data.shape[0]
print("sample", num_of_samples)
labels = np.ones((num_of_samples,), dtype='int64')
labels[0:7721] = 0
labels[7722:13433] = 1
labels[13434:20224] = 2
labels[20225:] = 3
names = ['epithelial', 'fibroblast', 'inflammatory', 'others']

Y = np_utils.to_categorical(labels, num_classes)
x, y = shuffle(img_data, Y, random_state=2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
img_height, img_width, channel = x_train.shape[1],x_train.shape[2],x_train.shape[3]
# input = Input(shape=(img_height, img_width, channel,))


# Conv2D_1 = Conv2D(64, (3,3), activation='relu', padding='same')(input)
# MaxPool2D_1 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(Conv2D_1)
# BatchNorm_1 = BatchNormalization()(MaxPool2D_1)


# tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(BatchNorm_1)
# tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)

# tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(BatchNorm_1)
# tower_2 = Conv2D(16, (5, 5), padding='same', activation='relu')(tower_2)

# tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(BatchNorm_1)
# tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(tower_3)

# output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

# Conv2D_2 = Conv2D(128, (3,3), activation='relu', padding='same')(output)
# MaxPool2D_2 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(Conv2D_2)
# BatchNorm_2 = BatchNormalization()(MaxPool2D_2)


# tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(BatchNorm_2)
# tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)

# tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(BatchNorm_2)
# tower_2 = Conv2D(16, (5, 5), padding='same', activation='relu')(tower_2)

# tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(BatchNorm_2)
# tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(tower_3)

# output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

# Conv2D_3 = Conv2D(256, (3,3), activation='relu', padding='same')(output)
# MaxPool2D_3 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(Conv2D_3)
# BatchNorm_3 = BatchNormalization()(MaxPool2D_3)

# tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(BatchNorm_3)
# tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)

# tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(BatchNorm_3)
# tower_2 = Conv2D(16, (5, 5), padding='same', activation='relu')(tower_2)

# tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(BatchNorm_3)
# tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(tower_3)

# output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

# Output = Flatten()(output)
# Output = Dense(num_classes, activation='softmax')(Output)
from keras.layers import Lambda,concatenate,GlobalAveragePooling2D

input = Input(shape=(224, 224, channel,))

def group_conv(input_tensor, num_filters, stride, n_groups):
    input_channels = K.int_shape(input_tensor)[-1]
    assert input_channels % n_groups == 0
    group_size = input_channels // n_groups

    conv_list = []
    for i in range(n_groups):
        offset = i * group_size
        group = Lambda(lambda x: x[:, :, :, offset:offset+group_size])(input_tensor)
        conv = Conv2D(num_filters // n_groups, (1, 1), strides=(1, 1), padding='same', activation='relu')(group)
        conv = Conv2D(num_filters // n_groups, (3, 3), strides=(stride, stride), padding='same', activation='relu')(conv)
        conv_list.append(conv)

    concat = concatenate(conv_list, axis=-1)
    return concat

# Stage 1
conv1 = Conv2D(24, (3, 3), strides=(2, 2), padding='same', activation='relu')(input)
maxpool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)

# Stage 2
group1 = group_conv(maxpool1, 24, stride=1, n_groups=3)
group2 = group_conv(group1, 48, stride=2, n_groups=3)

# Stage 3
group3 = group_conv(group2, 96, stride=1, n_groups=3)
maxpool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(group3)

# Stage 4
group4 = group_conv(maxpool3, 192, stride=2, n_groups=3)
group5 = group_conv(group4, 384, stride=1, n_groups=3)

# Stage 5
global_pooling = GlobalAveragePooling2D()(group5)
output = Dense(num_classes, activation='softmax')(global_pooling)

model = Model(inputs=[input], outputs=[output])
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy',f1,specificity,sensitivity])
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
csv_logger = CSVLogger('/content/drive/MyDrive/MINOR_PROJECT/shuffle_224.csv', append=True, separator=';')

data_augmentation = True

import time
start = time.time()

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,callbacks=[lr_reducer, csv_logger])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)  # randomly flip images
    datagen.fit(x_train)
    history = model.fit_generator(datagen.flow(x_train, y_train,
                                            batch_size=batch_size),
                               epochs=epochs,
                               validation_data=(x_test, y_test),callbacks=[lr_reducer,csv_logger])

print('------------Training time is seconds:%s',time.time()-start)
scores = model.evaluate(x_test, y_test, verbose=1)
print(scores)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print('Test f1:',scores[2])
print('Test specificity:',scores[3])
print('Test sensitivity:',scores[4])


import matplotlib.pyplot as plt
print("Max Test accuracy", max(history.history['val_accuracy']))
import matplotlib.pyplot  as plt

print(history.history.keys())
print(history.history.values())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("/content/drive/MyDrive/MINOR_PROJECT/shuffle_acc_224.png")
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("/content/drive/MyDrive/MINOR_PROJECT/shuffle_loss_224.png")
plt.show()

