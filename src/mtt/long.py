import pandas as pd
from time import perf_counter
from ..utils import Helper
from .models import MTTModel


class MTTLongList:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def train_list(self, fraction: float, epochs: int) -> None:
        train_df, test_df = Helper.split_data(self.df, fraction)

        train_gen, test_gen, train_img, val_img, test_img = Helper.create_generator(train_df, test_df)

        # Fit the models
        for name, model in MTTModel.models.items():
            # Get the model
            m = Helper.get_model(model['model'])
            MTTModel.models[name]['model'] = m

            start = perf_counter()

            # Fit the model
            history = m.fit(train_img, validation_data=val_img, epochs=epochs, verbose=1)

            # Sav the duration and the val_accuracy
            duration = perf_counter() - start
            duration = round(duration, 2)
            MTTModel.models[name]['perf'] = duration

            val_acc = history.history['val_accuracy']
            MTTModel.models[name]['val_acc'] = [round(v, 4) for v in val_acc]

        # Create a DataFrame with the results
        models_result = []

        for name, v in MTTModel.models.items():
            ratio = MTTModel.models[name]['val_acc'][-1] / MTTModel.models[name]['perf']
            models_result.append([name, MTTModel.models[name]['val_acc'][-1], MTTModel.models[name]['perf'], ratio])

        df_results = pd.DataFrame(models_result, columns=['model', 'val_accuracy', 'Training time (sec)', 'ratio'])
        df_results['ratio_norm'] = (df_results['ratio'] - df_results['ratio'].min()) / (
                df_results['ratio'].max() - df_results['ratio'].min())
        df_results.sort_values(by='val_accuracy', ascending=False, inplace=True)
        df_results.reset_index(inplace=True, drop=True)

        # Plot Accuracy
        Helper.plot_long_accuracy(df_results, epochs)

        # Plot Training Time
        Helper.plot_long_training_time(df_results)

        # Resort by normalized ratio
        df_results.sort_values(by='ratio_norm', ascending=False, inplace=True)
        df_results.reset_index(inplace=True, drop=True)

        # Print results to mention in paper
        for idx, row in df_results.iterrows():
            print(
                f"{row['model']} trained in {row['Training time (sec)']} sec, accuracy was {row['val_accuracy']}. Ratio: {round(row['ratio'], 2)}, normalized Ratio: {round(row['ratio_norm'], 2)}")
