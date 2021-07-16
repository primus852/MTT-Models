from src import Helper
from src import MTTLongList
from src import MTTShort
from src import OpenCVStream
from src import Prediction
import argparse

parser = argparse.ArgumentParser(description='Customize the training.')

# OpenCV Output
parser.add_argument('--production', help='Run only OpenCV with a specified Model', action='store_true', default=False)

# Params for LongList Training
parser.add_argument('--skip-long', help='Skip the Training of the Long Model-List', action='store_true', default=False)
parser.add_argument('--long-epochs', help='How many epochs should each Model on the LongList be trained',
                    action='store', type=int, default=5)
parser.add_argument('--long-fraction', help='Fraction of the Original Dataset to take for model evaluation',
                    action='store', type=float, default=0.05)

# Params for ShortList Training
parser.add_argument('--skip-short', help='Skip the Training of the Short Model-List', action='store_true',
                    default=False)
parser.add_argument('--short-epochs', help='How many epochs should each Model on the ShortList be trained',
                    action='store', type=int, default=10)

# Params for Plotting (Preprocess)
parser.add_argument('--skip-analysis', help='Skip the Plotting of the Dataset', action='store_true', default=False)

if __name__ == '__main__':

    # Get args from CLI
    args = parser.parse_args()

    # Create the folder to hold the results and plots
    result_folder, plot_folder = Helper.create_needed_folders()

    # Check if the dataset exists
    Helper.check_dataset()

    # Load the raw Data
    dataset = Helper.load_training_folder()

    if args.production:
        # ocv = OpenCVStream('MobileNet_06-07-2021_18_03_11')
        pred = Prediction()

        # Plot the times for the Short Model
        pred.plot_times()

        predictions = pred.make_predictions(dataset)


    else:

        # Optional, save/output a some Plots
        if not args.skip_analysis:
            Helper.plot_sample_distribution(df=dataset)
            Helper.plot_sample_images(df=dataset)

        # Optional, do the "LongList"
        if not args.skip_long:
            long = MTTLongList(dataset)
            long.train_list(fraction=args.long_fraction, epochs=args.long_epochs)

        # Train the Top X "ShortList"
        if not args.skip_short:
            short = MTTShort(dataset)
            short.train_list(epochs=args.short_epochs)
