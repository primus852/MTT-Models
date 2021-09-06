import tensorflow as tf
import matplotlib.pyplot as plt
import os
from pathlib import Path
import seaborn as sns
import time
import pandas as pd
from ..utils import Helper


class Prediction:

    @staticmethod
    def plot_times():
        df = pd.read_csv(
            Path(Helper.get_project_root() / Helper.RESULT_FOLDER / Helper.CSV_FOLDER / 'model_list.csv'),
            usecols=['model', 'avg', 'acc'])

        df['model'] = df['model'].apply(lambda x: x.split('_')[0])

        # Create a plot
        sns.barplot(x='model', y='avg', data=df)
        plt.title('Avg. Inference in seconds (batch = 32)', fontsize=15)
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.show()

    @staticmethod
    def make_predictions(df: pd.DataFrame):

        # Load models from the results folder
        model_folders = [f for f in
                         os.listdir(Path(Helper.get_project_root() / Helper.RESULT_FOLDER / Helper.MODEL_FOLDER)) if
                         not f.startswith('.') and not f.endswith('.txt') and not f.startswith('saved')]

        # Create a list of times for each model to make a prediction
        times = []

        # Load the Test Images
        train_df, test_df = Helper.split_data(df)
        train_gen, test_gen, train_img, val_img, test_imgs = Helper.create_generator(train_df, test_df)

        # Load the tensorflow models
        index = 0
        for model_path in model_folders:

            model_name = model_path.split('_')[0]
            m = tf.keras.models.load_model(
                Path(Helper.get_project_root() / Helper.RESULT_FOLDER / Helper.MODEL_FOLDER) / ('saved_' + model_path))

            run = 0
            total_prediction_time = 0
            for img in test_imgs:
                # Count up the number of runs
                run += 1

                # Time the prediction
                start_prediction = time.time()

                # Predict the class of the image
                m.predict(img)

                # Stop the timer
                end_prediction = time.time()

                # Add the time difference between start and stop to the total prediction time
                total_prediction_time += end_prediction - start_prediction

            # Calculate the prediction time for the model
            prediction_time = total_prediction_time / run

            # Create a new dict for the model with the prediction time as the value
            times.append({'Model': model_name, 'Avg. Prediction Time': prediction_time})
            print('Model' + model_name + ', avg. prediction Time was ' + str(prediction_time) + ' on ' + str(
                prediction_time) + ' images')

            index += 1

        return times
