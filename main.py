import os
from glob import glob
from PIL import Image
from keras.src.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.src.models import Sequential
from keras.src.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.src.optimizers import Adam
from keras.src.callbacks import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.src.utils.vis_utils import plot_model


def preprocessing_data(df):
    # Cleaning the dataset, there is null values in the age column
    df.fillna({'age': np.mean(df['age'])})

    base_skin_dir = './archive'
    # Merge image from both folders into one dictionary
    paths = {os.path.splitext(os.path.basename(x))[0]: x for x in
             glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

    # Give new column for the table for better understanding
    df['path'] = df['image_id'].map(paths.get)

    # Get the legible lesion type from the symbol
    lesion_type = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    # Give new column for the table for better understanding
    df['cell_type'] = df['dx'].map(lesion_type.get)
    df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes
    df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize((100, 75))))
    print(df.head())
    return paths, df


def DCNN_Model(df):
    features = df.drop(columns=['cell_type_idx'], axis=1)
    target = df['cell_type_idx']

    # Image process for features and the targets
    features, target = image_preprocessing(features, target)

    # Splitting into testing, training, and validating Xs and Ys
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=1234)
    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.1, random_state=2)

    input_shape = (75, 100, 3)
    num_classes = 7

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same', ))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='Same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='Same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.40))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    # Define the optimizer
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # Compile the model
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Set a learning rate annealer
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                learning_rate=0.00001)

    datagen = datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(x_train)

    # Fitting the model
    epochs = 50
    batch_size = 10
    history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_validate, y_validate),
                        verbose=1,
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        callbacks=[learning_rate_reduction])

    model.save("model.h5")

    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    loss_v, accuracy_v = model.evaluate(x_validate, y_validate, verbose=1)
    print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
    print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


def image_preprocessing(features, target):
    # Image Normalization
    features = np.asarray(features['image'].tolist())

    features_mean = np.mean(features)
    features_std = np.std(features)

    features = (features - features_mean) / features_std

    # One Hot Encoding
    target = to_categorical(target, num_classes=7)

    # Reshape image in 3 dimensions (height = 75px, width = 100px , canal = 3)
    features = features.reshape(features.shape[0], *(75, 100, 3))

    print(features, target)
    return features, target


def main():
    # Load the metadata
    skin_df = pd.read_csv('./archive/HAM10000_metadata.csv')

    image_id_path_dict, skin_df = preprocessing_data(skin_df)
    # print(skin_df.head())
    DCNN_Model(skin_df)


if __name__ == "__main__":
    main()
