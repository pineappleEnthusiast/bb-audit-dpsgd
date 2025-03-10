import numpy as np
from utils.data import load_data
import torch

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


def get_clipbkd(X, device):
    # calculate PCA of X
    flat_X = torch.flatten(X, start_dim=1) # N X 1 x D1 x D2 => N x (D1 * D2)
    trn_x = flat_X.cpu().numpy()
    n_comps = min(trn_x.shape[0], trn_x.shape[1]) # are there less data points or less pixels
    pca = PCA(n_comps) # get the top k components in the input space
    pca.fit(trn_x) # project the flattened data onto the top k components

    # choose vector associated with least singular value and scale it up to have same norm as the average
    avg_X_norm = torch.mean(torch.norm(flat_X, dim=1)) # whats the average norm (magnitude) of each flattened image
    target_X = avg_X_norm * torch.from_numpy(pca.components_[-1:]).to(device) # multiply the PC corresponding to smallest singular value by the avg norm
    target_X = torch.unsqueeze(target_X.reshape(X.shape[1:]), dim=0) # reshape that back into an image

    return target_X


device = 'cpu'
X_out, y_out, _ = load_data('mnist', -1, device=device)
target_X = get_clipbkd(X_out, device)

for k in range(10):
    target_y = torch.from_numpy(np.array([k])).to(device)
    X_in, y_in = torch.vstack((X_out, target_X)), torch.cat((y_out, target_y))

    X_in = X_in[y_in == k]
    y_in = y_in[y_in == k]

    # flatten each image to get a N x 784 vector
    flat_X = torch.flatten(X_in, start_dim=1)
    trn_x = flat_X.cpu().numpy()

    # PCA on images of class k
    pca = PCA(2)
    projected_data = pca.fit_transform(trn_x) # project the flattened data onto the top 2 components

    # visualize PCA
    colors = ['blue'] * (len(projected_data) - 1) + ['red']

    # Plot the projection
    plt.figure(figsize=(8, 6))
    plt.scatter(projected_data[:, 0], projected_data[:, 1], c=colors, alpha=0.7)
    plt.title(f"Projection onto First Two Principal Components: Class {k}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)

    plt.savefig('input_space.png')

    input('New Img')

    plt.close()
    

# try clipbkd for a fixed class


