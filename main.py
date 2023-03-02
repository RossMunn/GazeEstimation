import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set the image dimensions and batch size
img_height = 100
img_width = 100
batch_size = 32

# Define the model architecture
model = keras.Sequential([
    layers.Input(shape=(img_height, img_width, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Load the data and perform data augmentation
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_ds = train_datagen.flow_from_directory(
    'C:\\Users\\jrmun\\Desktop\\Dataset',
    target_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=123,
    subset='training'
)

val_ds = train_datagen.flow_from_directory(
    'C:\\Users\\jrmun\\Desktop\\Dataset',
    target_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=123,
    subset='validation'
)

# Train the model
history = model.fit(
    train_ds,
    epochs=70,
    validation_data=val_ds,
    verbose=2
)

# Save the model
model.save("my_model")





