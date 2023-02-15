import tensorflow as tf
from tensorflow import keras
tf.keras.models.Sequential()
from tensorflow.keras import layers

img_height = 90
img_width = 90
batch_size = 32


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'C:\\Users\\jrmun\Desktop\\Dataset',
    labels = 'inferred',
    label_mode = "int",
    class_names = ["close_look", "forward_look", "left_look", "right_look"],
    color_mode = 'grayscale',
    batch_size = batch_size,
    image_size = (img_height, img_width),
    seed = 123,
    validation_split = 0.1,
    subset = "training",
)

# Split the training dataset into training and validation subsets
val_ds = train_ds.take(int(len(train_ds) * 0.1))
train_ds = train_ds.skip(int(len(train_ds) * 0.1))


model = keras.Sequential([
    layers.Input((90, 90, 1)),
    layers.Conv2D(16, 3, padding='same'),
    layers.Conv2D(32, 3, padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10),
])


model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[
        keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    ],
    metrics=["accuracy"],
)

history = model.fit(
    train_ds,
    epochs=100,
    validation_data=val_ds,
    verbose=2
)




