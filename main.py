import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import wandb
import warnings


def display_multiple_samples(rows: int, cols: int) -> None:
    main_folder = os.getcwd()
    main_folder = main_folder + '\\Rice_Image_Dataset'
    types_of_rice = list(os.listdir(main_folder))
    types_of_rice.remove('Rice_Citation_Request.txt')
    for rice in types_of_rice:
        rice_folder = rice
        if rice == 'Basmati':
            rice = 'basmati'
        figure, ax = plt.subplots(rows, cols, figsize=(20, 10))
        plt.suptitle(rice.upper())
        for n in range(cols*rows):
            rice_image_name = f'{rice} ({int(np.random.randint(10000, 10100, 1))}).jpg'
            rice_image_path = os.path.join(main_folder, rice_folder, rice_image_name)
            image = plt.imread(rice_image_path)
            ax.ravel()[n].imshow(image)
            ax.ravel()[n].set_axis_off()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    display_multiple_samples(5, 5)
