import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


DATASET_PATH = "dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123
EPOCHS = 10


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names

print("\nClass:")
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
])


base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False


inputs = tf.keras.Input(shape=(224,224,3))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)


model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

model.summary()


os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

checkpoint = ModelCheckpoint(

    "models/mobilenet.keras",

    save_best_only=True,

    monitor="val_accuracy"

)

earlystop = EarlyStopping(

    patience=3,

    restore_best_weights=True

)


history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=[checkpoint, earlystop]

)


plt.figure(figsize=(10,4))

plt.subplot(1,2,1)

plt.plot(history.history["accuracy"])

plt.plot(history.history["val_accuracy"])

plt.title("Accuracy")

plt.legend(["Train","Validation"])

plt.subplot(1,2,2)

plt.plot(history.history["loss"])

plt.plot(history.history["val_loss"])

plt.title("Loss")

plt.legend(["Train","Validation"])

plt.tight_layout()

plt.savefig("results/mobilenet_training_result.png")

plt.show()

print("\nTraining selesai!")

print("Model tersimpan di models/mobilenet.keras")