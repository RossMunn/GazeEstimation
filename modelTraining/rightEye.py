import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns

def create_eye_gaze_model():
    model = models.Sequential()

    model.add(layers.Conv2D(24, (7, 7), activation='relu', padding='same', input_shape=(42, 50, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

    return model

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()

NUM_CLASSES = 7
data_dir = 'C:\\Users\\jrmun\Desktop\\train_right'
test_dir = 'C:\\Users\\jrmun\Desktop\\test_right'
BATCH_SIZE = 32
EPOCHS = 80
target_size = (42, 50)

data_generator = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    #horizontal_flip=True,
    #vertical_flip=True,
    brightness_range=(0.8, 1.2),  # Add brightness_range
    fill_mode='nearest'
)

train_generator = data_generator.flow_from_directory(
    data_dir,
    target_size=target_size,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale'
)

eye_gaze_model = create_eye_gaze_model()

eye_gaze_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

eye_gaze_model.summary()

#Add the ModelCheckpoint callback
model_checkpoint_callback = ModelCheckpoint(
filepath='C:\\Users\\jrmun\\PycharmProjects\\Disso\\Models\\best_eye_gaze_model_right.h5',
save_best_only=True,
monitor='val_accuracy',
mode='max',
verbose=1
)

history = eye_gaze_model.fit(
train_generator,
steps_per_epoch=train_generator.samples // BATCH_SIZE,
validation_data=validation_generator,
validation_steps=validation_generator.samples // BATCH_SIZE,
epochs=EPOCHS,
callbacks=[model_checkpoint_callback] # Add the callback here
)

plot_training_history(history)

#Load the saved model
loaded_model = load_model('C:\\Users\\jrmun\\PycharmProjects\\Disso\\Models\\best_eye_gaze_model_right.h5')

#Load the test data using a separate ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
test_dir,
target_size=(42, 50),
batch_size=BATCH_SIZE,
class_mode='categorical',
color_mode='grayscale',
shuffle=False
)

#Make predictions on the test data
y_pred = loaded_model.predict(test_generator)

#Load the true class labels for the test data
y_true = test_generator.classes

#Convert the predicted probabilities to class labels
import numpy as np
y_pred_labels = np.argmax(y_pred, axis=1)

#Print the classification report
from sklearn.metrics import classification_report
target_names = test_generator.class_indices.keys()
print(classification_report(y_true, y_pred_labels, target_names=target_names))



# Compute the confusion matrix
conf_mat = confusion_matrix(y_true, y_pred_labels)

# Normalize the confusion matrix
conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

# Plot the confusion matrix using a heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(conf_mat_normalized, annot=True, cmap='Blues', fmt='.2f', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
