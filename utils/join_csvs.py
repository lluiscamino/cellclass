from argparse import ArgumentParser

import pandas as pd

parser = ArgumentParser()
parser.add_argument("-o", "--output", dest="output_file",
                    help="write result CSV to FILE", metavar="FILE",
                    default="../data/train.csv")
parser.add_argument("-t", "--type", dest="type",
                    help="train or test", metavar="TYPE",
                    default="train")
args = parser.parse_args()

delimiter = "," if args.type == "train" else ";"

info_data = pd.read_csv(f"../data/{args.type}/info_{args.type}.csv", delimiter=";", decimal=",")
color_data = pd.read_csv(f"../data/{args.type}/color_{args.type}.csv", delimiter=delimiter, decimal=",") \
    .drop(columns="Class")
shape_data = pd.read_csv(f"../data/{args.type}/shape_{args.type}.csv", delimiter=delimiter, decimal=",")
texture_data = pd.read_csv(f"../data/{args.type}/texture_{args.type}.csv", delimiter=delimiter, decimal=",") \
    .drop(columns="Class")

data = info_data.join(color_data).join(shape_data).join(texture_data)
data.to_csv(args.output_file, index=False)
