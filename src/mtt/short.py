from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from .models import MTTModel
from ..utils import Helper
from sklearn.metrics import accuracy_score


class MTTShort:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def train_list(self, epochs: int, show_only: bool) -> None:
        train_df, test_df = Helper.split_data(self.df, 0.01)

        train_gen, test_gen, train_img, val_img, test_img = Helper.create_generator(train_df, test_df)

        # Fit the models
        for name, model in MTTModel.top_models.items():
            # Create Model folder
            now = datetime.now()
            model_name = '{}_{}'.format(name, now.strftime("%d-%m-%Y_%H_%M_%S"))
            checkpoint_path = Helper.create_model_checkpoints_folder(model_name)

            # Get the model
            m = Helper.get_model(model['model'])
            MTTModel.top_models[name]['model'] = m

            # Create a callback that saves the model's weights
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1)

            # Train the model
            history = m.fit(train_img,
                            validation_data=val_img,
                            epochs=epochs,
                            callbacks=[
                                tf.keras.callbacks.EarlyStopping(
                                    monitor='val_loss',
                                    patience=1,
                                    restore_best_weights=True), cp_callback])

            # Plot Comparison
            Helper.plot_loss_val_accuracy_comparison(history, model_name, show_only)

            # Predict the label of the test_images
            pred = model.predict(test_img)
            pred = np.argmax(pred, axis=1)

            # Map the label
            labels = (train_img.class_indices)
            labels = dict((v, k) for k, v in labels.items())
            pred = [labels[k] for k in pred]

            # Get the accuracy on the test set
            y_test = list(test_df.Label)
            acc = accuracy_score(y_test, pred)

            # Display the results
            print(f'## Best Model: {name} with {acc * 100:.2f}% accuracy on the test set')

            # Plot Confusion Matrix
            Helper.plot_confusion_matrix(y_test, pred, name, show_only)

            # Plot sample Predictions
            Helper.plot_prediction_sample(test_df, pred, name, show_only)
