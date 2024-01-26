import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

def dataset_to_dataframe(dataset):
    images = []
    labels = []
    for i in range(len(dataset)):
        img, label = dataset[i]
        # Remove the channel dimension and flatten the image
        img_flat = torch.flatten(img[0])  # img[0] converts from [1, 28, 28] to [28, 28]
        images.append(img_flat.numpy())
        labels.append(label)
    df = pd.DataFrame(images)
    df['label'] = labels
    return df


# Convert datasets to DataFrames
train = dataset_to_dataframe(training_data)
test = dataset_to_dataframe(test_data)

# Display the first few rows of the training DataFrame
# train_df.head()

# save the labels to a Pandas series target
target = train['label']
# Drop the label feature
train = train.drop("label",axis=1)

from sklearn.preprocessing import StandardScaler
X = train.values
X_std = StandardScaler().fit_transform(X)

# Calculating Eigenvectors and eigenvalues of Cov matirx
mean_vec = np.mean(X_std, axis=0)
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# Create a list of (eigenvalue, eigenvector) tuples
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the eigenvalue, eigenvector pair from high to low
eig_pairs.sort(key = lambda x: x[0], reverse= True)

# Calculation of Explained Variance from the eigenvalues
tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance

# Invoke SKlearn's PCA method
n_components = 30
pca = PCA(n_components=n_components).fit(train.values)

eigenvalues = pca.components_.reshape(n_components, 28, 28)

# Extracting the PCA components ( eignevalues )
#eigenvalues = pca.components_.reshape(n_components, 28, 28)
eigenvalues = pca.components_

n_row = 4
n_col = 7

# Plot the first 8 eignenvalues
# plt.figure(figsize=(13,12))
# for i in list(range(n_row * n_col)):
#     offset =0
#     plt.subplot(n_row, n_col, i + 1)
#     plt.imshow(eigenvalues[i].reshape(28,28), cmap='jet')
#     title_text = 'Eigenvalue ' + str(i + 1)
#     plt.title(title_text, size=6.5)
#     plt.xticks(())
#     plt.yticks(())
# plt.show()

# Delete our earlier created X object
del X
# Taking only the first N rows to speed things up
X= train[:6000].values
target_x = target[0:6000]
del train
# Standardising the values
X_std = StandardScaler().fit_transform(X)

# Call the PCA method with 5 components. 
pca = PCA(n_components=5)
pca.fit(X_std)
X_5d = pca.transform(X_std)



import seaborn as sns

# We have X_5d which are the first five principal components
# We'll use only the first two for the scatter plot
X_pca = X_5d[:, :2]

# Create a scatter plot
plt.figure(figsize=(10, 8))
palette=sns.color_palette("hsv", 10)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=target_x, palette=sns.color_palette("hsv", 10))

# Setting titles and labels
plt.title('Scatter Plot of First Two Principal Components')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

# Adding legend
plt.legend(title='Fashion Item', labels=list(labels_map.values()), bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.show()