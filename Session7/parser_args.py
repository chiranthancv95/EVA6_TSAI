import argparse

parser = argparse.ArgumentParser()


parser.add_argument('-e','--epochs', type=int, default=20, help="Number of epochs to train")
args = parser.parse_args()

norm = args.norm
epochs = args.epochs