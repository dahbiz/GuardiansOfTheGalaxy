import sys 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

saving_path = "proccessed_data/"
cropped_galaxy_imgs = np.load(saving_path + "cropped_galaxy_imgs.npy", allow_pickle=True).item()


# transform
def our_PCA(data_dict, k):
    print("Performing PCA ...")
    pca_galaxy_imgs = {}
    inv_pca_galaxy_imgs = {}
    pca = PCA(n_components=k, svd_solver='full', random_state=42)
    count = 1
    total = len(data_dict)
    for key, val in data_dict.items():
        if count == total + 1:
            break
        sys.stdout.write("\r" + str(count) + " / " + str(total))
        sys.stdout.flush()
        pca_galaxy_imgs[key] = pca.fit_transform(val)
        inv_pca_galaxy_imgs[key] = pca.inverse_transform(pca_galaxy_imgs[key])
        count += 1
    print("\ndone!")
    return  pca_galaxy_imgs, inv_pca_galaxy_imgs



# running pca
pca_galaxy_imgs, inv_pca_galaxy_imgs = our_PCA(cropped_galaxy_imgs, 0.9)

# saving pca and inv pca arrays
print("Saving pca_galaxy_imgs ...\n")
np.save(saving_path + "pca_galaxy_imgs.npy", pca_galaxy_imgs)
print("\ndone!")

print("\nSaving inv_pca_galaxy_imgs ...")
np.save(saving_path + "inv_pca_galaxy_imgs.npy", inv_pca_galaxy_imgs)
print("\ndone!")


