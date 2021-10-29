import os
import PIL
import pandas as pd
import numpy as np
from pandas.io import json
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.image as mpimg

# directory paths
PROJECT_PATH = Path().resolve().parent
DATA_PATH = PROJECT_PATH.joinpath('data')

# Total number of images
print("\nTotal Images:", len(list(DATA_PATH.glob("*/*.jpg"))), end="\n\n")

# read training csv file
df = pd.read_csv(DATA_PATH.joinpath("train.csv"))
print(df.head(), end="\n\n")

# Unique classes in the dataset
with open(DATA_PATH.joinpath("label_num_to_disease_map.json")) as file:
    print("\nImage classes", json.dumps(
        json.loads(file.read()), indent=4), end="\n\n")

# Bar chart of disease labels
df['label'].value_counts().plot.bar()
plt.show()

# randomly select images from healthy and disease plants and plot them\

cmd_sample = df[df['label'] == 3].sample(6)
healthy_samples = df[df['label'] == 4].sample(6)

for i, samples in enumerate([cmd_sample, healthy_samples]):
    if i == 0:
        title = "CMD Disease Sample"
    else:
        title = "Healthy Samples"
    fig = plt.figure(figsize=(12, 8))
    for index, image_id in enumerate(samples['image_id']):
        plt.subplot(3, 3, index+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.title("{}".format(title))
        plt.imshow(mpimg.imread(DATA_PATH.joinpath(
            'train_images', "{}".format(image_id))))
    plt.show()
