from datetime import datetime
import pandas as pd
from pathlib import Path
import numpy as np
import tensorflow as tf
from .models import MTTModel
from ..utils import Helper
from sklearn.metrics import accuracy_score
import time
import csv


class MTTShort:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def train_list(self, epochs: int) -> None:
        train_df, test_df = Helper.split_data(self.df)

        train_gen, test_gen, train_img, val_img, test_img = Helper.create_generator(train_df, test_df)

        # Create a list of times for each model to make a prediction
        times = []

        # Fit the models
        for name, model in MTTModel.top_models.items():
            # Create Model folder
            now = datetime.now()
            model_name = '{}_{}_epochs_{}'.format(name, epochs, now.strftime("%d-%m-%Y_%H_%M_%S"))
            checkpoint_path = Helper.create_model_checkpoints_folder(model_name)

            # Get the model
            m = Helper.get_model(model['model'])
            MTTModel.top_models[name]['model'] = m

            # Log dir (gets created)
            log_dir = "logs/" + model_name

            # Create a callback that saves the model's weights
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1)

            # Callback for Tensorboard
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            # Callback for EarlyStopping
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1,
                                                                       restore_best_weights=True)

            # Train the model
            history = m.fit(train_img,
                            validation_data=val_img,
                            epochs=epochs,
                            callbacks=[cp_callback, tensorboard_callback, early_stopping_callback])

            m.save('results/models/saved_' + model_name)

            # Plot Comparison
            Helper.plot_loss_val_accuracy_comparison(history, model_name)

            # Time the prediction
            start_prediction = time.time()

            # Predict the label of the test_images
            pred = m.predict(test_img)

            # Stop the timer
            end_prediction = time.time()

            # Calculate the prediction time for the model
            prediction_time = (end_prediction - start_prediction) / len(test_img)

            pred = np.argmax(pred, axis=1)

            # Map the label
            labels = train_img.class_indices
            labels = dict((v, k) for k, v in labels.items())
            pred = [labels[k] for k in pred]

            # Get the accuracy on the test set
            y_test = list(test_df.Label)
            acc = accuracy_score(y_test, pred)

            # Display the results
            print(f'## Best Model: {name} with {acc * 100:.2f}% accuracy on the test set')

            # Create a new dict for the model with the prediction time as the value
            times.append({'model': model_name, 'avg': prediction_time, 'acc': acc})

            # Plot Confusion Matrix
            Helper.plot_confusion_matrix(y_test, pred, name)

            # Plot sample Predictions
            Helper.plot_prediction_sample(test_df, pred, name)

        with open(Path(Helper.get_project_root() / Helper.RESULT_FOLDER / Helper.CSV_FOLDER / 'model_list.csv'), 'w',
                  newline='', encoding='utf-8') as csvfile:
            fieldnames = ['model', 'avg', 'acc']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in times:
                writer.writerow(row)
