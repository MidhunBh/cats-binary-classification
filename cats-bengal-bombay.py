https://drive.google.com/drive/folders/1YWuwMbAKWRCpBUagnh55kWlYmEtRe23u

#Import libraries and dataset
"""

import tensorflow as tf
import pandas as pd
import numpy as np

IMAGE_SIZE=256
BATCH_SIZE=32
CHANNELS=3
EPOCHS=50

images = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/drive/MyDrive/cats binary classification',
    labels='inferred',

    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
)

class_names=images.class_names
class_names

"""#Exploring the dataset

"""

for image_batch,labels_batch in images.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())

"""#Print first image in the batch

"""

#Print first image in the batch
for image_batch,label_batch in images.take(1):
    print(image_batch[0].numpy())

import matplotlib.pyplot as plt
#Visualize the first image in that batch
for image_batch, labels_batch in images.take(1):
        plt.imshow(image_batch[0].numpy().astype("uint8"))
        plt.title(class_names[labels_batch[0]])
        plt.axis('off')

plt.figure(figsize=(6,6))
for image_batch, labels_batch in images.take(1):
    for i in range(4):
        ax = plt.subplot(2,2, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[labels_batch[i]])
        plt.axis("off")

len(images)

def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds=get_dataset_partitions_tf(images)

len(train_ds)

len(val_ds)

len(test_ds)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

resize_and_rescale = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
  tf.keras.layers.experimental.preprocessing.Rescaling(1.0/255),
])

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

from keras import models,layers

"""#Building the model

"""

input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3

model = models.Sequential([
    resize_and_rescale,
    layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='sigmoid'),
])

model.build(input_shape=input_shape)

"""#Summary of the model

"""

model.summary()

"""#Compiling the model

"""

adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

"""#Train the network with the given inputs and the corresponding labels

"""

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds,
)

"""#Evaluate the model

"""

scores = model.evaluate(test_ds)

scores

history

print(history.params)

print(history.history.keys())

history.history['accuracy']

len(history.history['accuracy'])

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

"""#Plotting Accuracy, Loss graph and Training and Validation Accuracy

"""

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS),acc,label='Training Accuracy')
plt.plot(range(EPOCHS),val_acc,label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(range(EPOCHS),loss,label='Training Loss')
plt.plot(range(EPOCHS),val_loss,label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

"""#Making predictions

"""

for images_batch,label_batch in test_ds.take(1):
    plt.imshow(images_batch[0].numpy().astype('uint8'))

for images_batch,label_batch in test_ds.take(1):
    image1=images_batch[0].numpy().astype('uint8')
    label1=label_batch[0].numpy()

    print('Predicting the first image ')
    plt.imshow(image1)
    print('image1 True Label ',class_names[label1])
    batch_prediction=model.predict(images_batch)
    print("image1's predicted label :",class_names[np.argmax(batch_prediction[0])])

"""#Function to predict with confidence

"""

def predict(model,img):
    img_array=tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array=tf.expand_dims(img_array,0)

    predictions=model.predict(img_array)
    predicted_class=class_names[np.argmax(predictions[0])]
    confidence=round(100*(np.max(predictions[0])),2)
    return predicted_class,confidence

plt.figure(figsize=(15,15))
for images,labels in test_ds.take(1):
    for i in range(9):
        ax=plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        predicted_class,confidence=predict(model,images[i].numpy())
        actual_class=class_names[labels[i]]
        plt.title(f"Actual:{actual_class},\n Predicted:{predicted_class},\n Confidence:{confidence}%")
        plt.axis("off")

"""#Saving the model"""

# model_version=1
model.save("/content/drive/MyDrive/cat")

model.save("/content/drive/MyDrive/cat/indiancats.h5")
