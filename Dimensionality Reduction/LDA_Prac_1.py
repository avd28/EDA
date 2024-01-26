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

lda = LDA(n_components=5)
# Taking in as second argument the Target as labels
X_LDA_2D = lda.fit_transform(X_std, target.values )

import seaborn as sns

# We have X_5d which are the first five principal components
# We'll use only the first two for the scatter plot
X_lda = X_LDA_2D[:, :2]

# Create a scatter plot
plt.figure(figsize=(10, 8))
palette=sns.color_palette("hsv", 10)
sns.scatterplot(x=X_lda[:, 0], y=X_lda[:, 1], hue=target, palette=sns.color_palette("hsv", 10))

# Setting titles and labels
plt.title('Scatter Plot of First and Second Linear Discriminants')
plt.xlabel('First Linear Discriminant')
plt.ylabel('Second Linear Discriminant')

# Adding legend
plt.legend(title='Fashion Item', labels=list(labels_map.values()), bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.show()