import pandas as pd
import numpy as np
labels = pd.read_csv("Galaxy_data/training_solutions_rev1.csv")
labels=labels.drop(labels[labels['Class1.3'] > 0.5].index)
labels = labels[['GalaxyID', 'Class1.1', 'Class1.2', 'Class6.1']]
labels['Result'] = 'i'
labels.loc[(labels['Class1.1'] > 0.7) & (labels['Class6.1'] < 0.1), 'Result'] = 'e'
labels.loc[(labels['Class1.2'] > 0.7) & (labels['Class6.1'] < 0.1), 'Result'] = 's'
labels.loc[labels['Class6.1'] > 0.63, 'Result'] = 'o'

elip_df = labels[labels['Result'] == 'e']
spiral_df = labels[labels['Result'] == 's']
odd_df = labels[labels['Result'] == 'o']

# Sample 5000 values from each category
category1_sampled = elip_df.sample(n=5000, random_state=42)
category2_sampled = spiral_df.sample(n=5000, random_state=42)
category3_sampled = odd_df.sample(n=5000, random_state=42)

# Concatenate the sampled DataFrames back together
sampled_df = pd.concat([category1_sampled, category2_sampled, category3_sampled])

galaxy =  sampled_df.sort_values(by='GalaxyID')
galaxy_ids = galaxy['GalaxyID'].to_numpy()
galaxy_dict = np.load("proccessed_data/cropped_galaxy_imgs.npy", allow_pickle=True).item()
flattened_dict = {}

for key, array in galaxy_dict.items():
    # Flatten the 2D array into a 1D array using numpy.ravel()
    flattened_array = np.ravel(array)
    
    # Update the dictionary with the flattened array
    flattened_dict[key] = flattened_array
    
#flattened_dict
ordered_dict = dict(sorted(flattened_dict.items(), key=lambda x: x[0]))
df = pd.DataFrame(ordered_dict).transpose()

df.columns = df.iloc[0]

#df = df[1:]

total_pixels = len(df.columns) 
header = [f'Pixel_{i}' for i in range(1, total_pixels + 1)]

df.columns = header
mask = df.index.isin(galaxy_ids)
df = df[mask]
new_column =galaxy['Result']
new_column = new_column.values.astype(str)
df.insert(0, 'Shape', new_column)

print("df done")
from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib 
# Assuming your DataFrame is named 'df' and the first column is the target
target_column = df.columns[0]
features = df.drop(target_column, axis=1)
category_mapping = {'e': 0, 's': 1, 'o': 2}
target = df[target_column].map(category_mapping)
print("starting pca")

pca=PCA(n_components=0.9)
f_reduced=pca.fit_transform(features)
with open("pca_fast.npy",'wb') as f:
    np.save(f,f_reduced)
print("pca done")
