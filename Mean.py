import pandas as pd
import numpy as np
labels = pd.read_csv("Galaxy_data/training_solutions_rev1.csv")
labels=labels.drop(labels[labels['Class1.3'] > 0.5].index)
labels = labels[['GalaxyID', 'Class1.1', 'Class1.2', 'Class6.1']]
labels['Result'] = 'i'
labels.loc[labels['Class1.1'] > 0.6, 'Result'] = 'e'
labels.loc[labels['Class1.2'] > 0.6, 'Result'] = 's'
labels.loc[labels['Class6.1'] > 0.6, 'Result'] = 'o'
labels=labels.drop(labels[labels['Result'] == 'i'].index)
galaxy_ids = labels['GalaxyID'].to_numpy()
from PIL import Image, ImageOps
from numpy import asarray
import os

folder_path = "Galaxy_data/images_training_rev1/"
output_dict = {}

for galaxy_id in galaxy_ids:
    filename = f"{galaxy_id}.jpg"
    filepath = os.path.join(folder_path, filename)

    if os.path.exists(filepath):
        # load the image and convert into numpy array
        img = Image.open(filepath)
        img_gray = ImageOps.grayscale(img)
        numpydata = asarray(img_gray).flatten()

        # store the numpy array in the dictionary with the galaxy_id as the key
        output_dict[galaxy_id] = numpydata

data_array = np.array(list(output_dict.values()))

mean_array = np.mean(data_array, axis=0)       
with open('mean.npy','wb') as f:
    np.save(f,mean_array)