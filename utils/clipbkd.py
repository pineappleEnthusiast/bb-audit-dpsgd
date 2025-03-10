"""Utility functions to craft ClipBKD target sample from https://github.com/jagielski/auditing-dpsgd
Paper: https://proceedings.neurips.cc/paper_files/paper/2020/file/fc4ddc15f9f4b4b06ef7844d6bb53abf-Paper.pdf"""
import torch
from sklearn.decomposition import PCA

def choose_worstcase_label(model, target_X):
    with torch.no_grad():
        # pick class maximizing gradient norm 
        output = model(target_X)
        target_y = torch.unsqueeze(torch.argmin(output), dim=0)
    
    return target_y
    
def craft_clipbkd(X, model, device='cpu'):
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

    target_y = choose_worstcase_label(model, target_X)
    
    return target_X, target_y