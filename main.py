from src import Helper
from src import MTTLongList
from src import MTTShort

if __name__ == '__main__':
    # Create the folder to hold the results and plots
    result_folder, plot_folder = Helper.create_needed_folders()

    # Load the raw Data
    dataset = Helper.load_training_folder()

    # Optional, save/output a some Plots
    Helper.plot_sample_distribution(df=dataset, show_only=False)
    Helper.plot_sample_images(df=dataset, show_only=False)

    # Optional, do the "LongList"
    long = MTTLongList(dataset)
    long.train_list(fraction=0.10, epochs=10, show_only=False)

    # Train the Top X "ShortList"
    # short = MTTShort(dataset)
    # short.train_list(epochs=1, show_only=False)
