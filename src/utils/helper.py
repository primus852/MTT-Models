import os
from pathlib import Path
from typing import Tuple, List
from .download import Downloader

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import zipfile


class Helper:
    RESULT_FOLDER: str = 'results'
    PLOT_FOLDER: str = 'plots'
    MODEL_FOLDER: str = 'models'
    OPENCV_FOLDER: str = 'video'
    DATASET_FOLDER: str = 'data'
    DATASET_GDRIVE_ID: str = '1batvXHflZy72ACJPnRfp5rxYkcRZosvY'
    DATASET_GDRIVE_NAME: str = 'asl.zip'

    @staticmethod
    def get_project_root() -> Path:
        return Path(__file__).parent.parent.parent

    @staticmethod
    def check_dataset():

        # Dataset Folder
        data_path = str(Helper.get_project_root() / Helper.DATASET_FOLDER)

        if not Path(data_path).is_dir():

            try:
                os.mkdir(data_path)
            except OSError:
                print("Creation of the directory %s failed" % data_path)

            print('### DATASET NOT FOUND ###')
            full_path = str(Helper.get_project_root() / Helper.DATASET_FOLDER / Helper.DATASET_GDRIVE_NAME)
            print('Downloading, to {}, please be patient...'.format(full_path))

            Downloader.download_file_from_google_drive(Helper.DATASET_GDRIVE_ID, full_path)

            print('Download completed, unzipping...')

            with zipfile.ZipFile(
                    str(Path(Helper.get_project_root() / Helper.DATASET_FOLDER / Helper.DATASET_GDRIVE_NAME)),
                    'r') as zip_ref:
                zip_ref.extractall(str(Path(Helper.get_project_root() / Helper.DATASET_FOLDER)))

            print('Dataset unzipped, continuing...')

            # Unlink the Zip
            os.remove(str(Path(Helper.get_project_root() / Helper.DATASET_FOLDER / Helper.DATASET_GDRIVE_NAME)))

        else:
            print('### DATASET FOUND ###')

    @staticmethod
    def create_needed_folders() -> Tuple[str, str]:

        # Training Folder
        result_path = str(Helper.get_project_root() / Helper.RESULT_FOLDER)
        if not Path(result_path).is_dir():
            try:
                os.mkdir(result_path)
            except OSError:
                print("Creation of the directory %s failed" % result_path)

        # Plot Folder
        plot_path = str(Helper.get_project_root() / Helper.RESULT_FOLDER / Helper.PLOT_FOLDER)
        if not Path(plot_path).is_dir():
            try:
                os.mkdir(plot_path)
            except OSError:
                print("Creation of the directory %s failed" % plot_path)

        # Model Folder
        model_path = str(Helper.get_project_root() / Helper.RESULT_FOLDER / Helper.MODEL_FOLDER)
        if not Path(model_path).is_dir():
            try:
                os.mkdir(model_path)
            except OSError:
                print("Creation of the directory %s failed" % model_path)

        # OpenCV Folder
        opencv_path = str(Helper.get_project_root() / Helper.RESULT_FOLDER / Helper.OPENCV_FOLDER)
        if not Path(opencv_path).is_dir():
            try:
                os.mkdir(opencv_path)
            except OSError:
                print("Creation of the directory %s failed" % opencv_path)

        return result_path, plot_path

    @staticmethod
    def create_model_checkpoints_folder(modelname: str) -> str:
        model_path = str(Helper.get_project_root() / Helper.RESULT_FOLDER / Helper.MODEL_FOLDER / modelname)
        if not Path(model_path).is_dir():
            try:
                os.mkdir(model_path)
            except OSError:
                print("Creation of the directory %s failed" % model_path)

        return model_path

    @staticmethod
    def load_training_folder() -> pd.DataFrame:
        """
        Load the raw data
        :return:
        """
        path = list((Helper.get_project_root() / Helper.DATASET_FOLDER / 'asl_alphabet_train').rglob('**/*.jpg'))
        files = [x for x in path if x.is_file()]

        labels = [str(files[i]).split("\\")[-2] for i in range(len(files))]

        filepath = pd.Series(files, name='Filepath').astype(str)
        labels = pd.Series(labels, name='Label')

        # Concatenate filepaths and labels
        df = pd.concat([filepath, labels], axis=1)

        # Shuffle the DataFrame and reset index
        df = df.sample(frac=1).reset_index(drop=True)

        # Create Labels .txt
        class_file = str(Helper.get_project_root() / Helper.RESULT_FOLDER / Helper.MODEL_FOLDER / 'classes.txt')
        with open(class_file, 'w') as f:
            for c in df.Label.unique():
                f.write(c + '\n')

        return df

    @staticmethod
    def plot_loss_val_accuracy_comparison(df: pd.DataFrame, model_name: str) -> None:

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        ax = axes.flat

        pd.DataFrame(df.history)[['accuracy', 'val_accuracy']].plot(ax=ax[0])
        ax[0].set_title("Accuracy", fontsize=15)
        ax[0].set_ylim(0, 1.1)

        pd.DataFrame(df.history)[['loss', 'val_loss']].plot(ax=ax[1])
        ax[1].set_title("Loss", fontsize=15)

        plt.savefig(
            str(Helper.get_project_root() / Helper.RESULT_FOLDER / Helper.PLOT_FOLDER / 'compare_{}.png'.format(
                model_name)))

    @staticmethod
    def plot_prediction_sample(test_df: pd.DataFrame, pred: List, model_name: str) -> None:

        fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(20, 12),
                                 subplot_kw={'xticks': [], 'yticks': []})

        for i, ax in enumerate(axes.flat):
            ax.imshow(plt.imread(test_df.Filepath.iloc[i]))
            ax.set_title(f"True: {test_df.Label.iloc[i].split('_')[0]}\nPredicted: {pred[i].split('_')[0]}",
                         fontsize=15)
        plt.tight_layout()

        plt.savefig(
            str(Helper.get_project_root() / Helper.RESULT_FOLDER / Helper.PLOT_FOLDER / 'sample_predictions_{}.png'.format(
                model_name)))

    @staticmethod
    def plot_confusion_matrix(y_test: List, pred: List, model_name: str) -> None:
        cf_matrix = confusion_matrix(y_test, pred, normalize='true')
        plt.figure(figsize=(17, 12))
        sns.heatmap(cf_matrix, annot=True, xticklabels=sorted(set(y_test)), yticklabels=sorted(set(y_test)), cbar=False)
        plt.title('Normalized Confusion Matrix', fontsize=23)
        plt.xticks(fontsize=12, rotation=45)
        plt.yticks(fontsize=12)

        plt.savefig(
            str(Helper.get_project_root() / Helper.RESULT_FOLDER / Helper.PLOT_FOLDER / 'confusion_matrix_{}.png'.format(
                model_name)))

    @staticmethod
    def plot_sample_distribution(df: pd.DataFrame) -> None:
        """
        Plot distribution of Training Data
        :return:
        """

        # Plot / Save Plot
        vc = df['Label'].value_counts()
        plt.figure(figsize=(20, 5))
        sns.barplot(x=sorted(vc.index), y=vc, palette="rocket")
        plt.title("Number of pictures of each category", fontsize=15)
        plt.savefig(str(Helper.get_project_root() / Helper.RESULT_FOLDER / Helper.PLOT_FOLDER / 'distribution.png'))

    @staticmethod
    def plot_long_accuracy(df: pd.DataFrame, epochs: int) -> None:
        plt.figure(figsize=(15, 5))
        sns.barplot(x='model', y='val_accuracy', data=df)
        plt.title('Accuracy on the test set (after {} epoch))'.format(epochs), fontsize=15)
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(
            str(Helper.get_project_root() / Helper.RESULT_FOLDER / Helper.PLOT_FOLDER / 'long_val_accuracy.png'))

    @staticmethod
    def plot_long_training_time(df: pd.DataFrame) -> None:
        plt.figure(figsize=(15, 5))
        sns.barplot(x='model', y='Training time (sec)', data=df)
        plt.title('Training time for each model in sec', fontsize=15)
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(
            str(Helper.get_project_root() / Helper.RESULT_FOLDER / Helper.PLOT_FOLDER / 'long_training_time.png'))

    @staticmethod
    def plot_sample_images(df: pd.DataFrame) -> None:
        """
        Plot a sample fo Images
        @TODO: Avoid duplicates (possibly with load_training_folder())
        :param df:
        :param show_only:
        :return:
        """

        fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(15, 7), subplot_kw={'xticks': [], 'yticks': []})

        for i, ax in enumerate(axes.flat):
            ax.imshow(plt.imread(df.Filepath[i]))
            ax.set_title(df.Label[i], fontsize=15)
        plt.tight_layout(pad=0.5)

        plt.savefig(str(Helper.get_project_root() / Helper.RESULT_FOLDER / Helper.PLOT_FOLDER / 'samples.png'))

    @staticmethod
    def create_generator(df_train: pd.DataFrame, df_test: pd.DataFrame):
        """
        Load the images with a generator and data augmentation
        :param df_train:
        :param df_test:
        :return:
        """

        train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
            validation_split=0.1
        )

        test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
        )

        train_images = train_generator.flow_from_dataframe(
            dataframe=df_train,
            x_col='Filepath',
            y_col='Label',
            target_size=(224, 224),
            color_mode='rgb',
            class_mode='categorical',
            batch_size=32,
            shuffle=True,
            seed=0,
            subset='training',
            rotation_range=30,  # Uncomment to use data augmentation
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        val_images = train_generator.flow_from_dataframe(
            dataframe=df_train,
            x_col='Filepath',
            y_col='Label',
            target_size=(224, 224),
            color_mode='rgb',
            class_mode='categorical',
            batch_size=32,
            shuffle=True,
            seed=0,
            subset='validation',
            rotation_range=30,  # Uncomment to use data augmentation
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        test_images = test_generator.flow_from_dataframe(
            dataframe=df_test,
            x_col='Filepath',
            y_col='Label',
            target_size=(224, 224),
            color_mode='rgb',
            class_mode='categorical',
            batch_size=32,
            shuffle=False
        )

        return train_generator, test_generator, train_images, val_images, test_images

    @staticmethod
    def get_model(model):
        kwargs = {'input_shape': (224, 224, 3),
                  'include_top': False,
                  'weights': 'imagenet',
                  'pooling': 'avg'}

        pretrained_model = model(**kwargs)
        pretrained_model.trainable = False

        inputs = pretrained_model.input

        x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
        x = tf.keras.layers.Dense(128, activation='relu')(x)

        outputs = tf.keras.layers.Dense(29, activation='softmax')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    @staticmethod
    def split_data(df: pd.DataFrame, fraction: float = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the Dataset in Train and Test
        :param df:
        :param fraction: Percentage of the original dataset
        :return:
        """
        train_df, test_df = train_test_split(df.sample(frac=fraction), test_size=0.2)

        return train_df, test_df
