import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import json


data_dir = 'data/Medicinal Leaf dataset'  # Update this path to your dataset location
img_size = 224
batch_size = 32
base_learning_rate = 1e-4
epochs = 30

def preprocess(img):
    return tf.keras.applications.mobilenet_v2.preprocess_input(img)

def random_grayscale(img):
    if tf.random.uniform(()) < 0.3:
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.grayscale_to_rgb(img)
    return img

def augmentation_fn(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k)
    img = random_grayscale(img)
    return img

def decode_img(img_bytes):
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32)
    return img

def load_and_preprocess(image_path, label):
    img_bytes = tf.io.read_file(image_path)
    img = decode_img(img_bytes)
    img = augmentation_fn(img)
    img = preprocess(img)
    return img, label

def load_and_preprocess_no_aug(image_path, label):
    img_bytes = tf.io.read_file(image_path)
    img = decode_img(img_bytes)
    img = preprocess(img)
    return img, label

def get_dataset(data_dir, batch_size, augment=False, shuffle=True, shuffle_buffer=1000):
    AUTOTUNE = tf.data.AUTOTUNE
    list_ds = tf.data.Dataset.list_files(str(data_dir + '/*/*'), shuffle=shuffle)

    list_ds = list_ds.filter(lambda x: tf.strings.regex_full_match(x, ".*(jpg|jpeg|png|bmp|gif|JPG|JPEG|PNG|BMP|GIF)"))

    class_names = np.array(sorted([item.name for item in os.scandir(data_dir) if item.is_dir()]))

    def get_label(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]  # folder name
        return tf.squeeze(tf.where(class_names == label)[0])

    labeled_ds = list_ds.map(lambda x: (x, get_label(x)), num_parallel_calls=AUTOTUNE)

    if augment:
        ds = labeled_ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    else:
        ds = labeled_ds.map(load_and_preprocess_no_aug, num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds, class_names

def main():
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found!")
        print("Please update the 'data_dir' variable in this script to point to your dataset location.")
        return
    
    print("Loading datasets...")
    train_ds, class_names = get_dataset(data_dir, batch_size, augment=True, shuffle=True)
    val_ds, _ = get_dataset(data_dir, batch_size, augment=False, shuffle=False)
    
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    print("Building model...")
    input_tensor = Input(shape=(img_size, img_size, 3))
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=input_tensor)

    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=base_learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_medicinal_leaf_model.h5', monitor='val_accuracy', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)
    tensorboard_callback = TensorBoard(log_dir='./logs')

    callback_list = [early_stopping, model_checkpoint, reduce_lr, tensorboard_callback]

    # Training
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callback_list,
        verbose=1
    )
    
    print("Saving model...")
    model.save('medicinal_leaf_model.h5')
    
    class_names_dict = {'class_names': class_names.tolist()}
    with open('class_names.json', 'w') as f:
        json.dump(class_names_dict, f)
    
    print("Training completed!")
    print(f"Model saved as 'medicinal_leaf_model.h5'")
    print(f"Class names saved as 'class_names.json'")
    
    
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Validation accuracy: {val_acc:.3f}")

if __name__ == "__main__":
    main()