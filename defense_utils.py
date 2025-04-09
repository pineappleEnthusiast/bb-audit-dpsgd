import numpy as np
import torch


def whiten(G_centered):
    cov_matrix = G_centered.T @ G_centered
    eigvals, eigvecs = torch.linalg.eigh(cov_matrix)
    eigvals = torch.clamp(eigvals, min=1e-6)  # Ensure stability
    W_pca = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T
    return W_pca

    # cov_matrix_inv = torch.linalg.inv(cov_matrix)
    # eigvals, eigvecs = torch.linalg.eigh(cov_matrix_inv)
    # eigvals = torch.clamp(eigvals, min=1e-5)
    # cov_matrix_inv = eigvecs @ torch.diag(eigvals) @ eigvecs.T
    # return torch.linalg.cholesky(cov_matrix_inv)


def _whiten(G_centered):
    cov_matrix = G_centered.T @ G_centered
    cov_matrix_inv = torch.linalg.inv(cov_matrix)

    # NOTE: see if this affects accuracy; intention is to account for numerical instability so cholesky decomposition can work; MNIST rankings still the same
    eigvals, eigvecs = torch.linalg.eigh(cov_matrix_inv)  # Compute eigenvalues & eigenvectors
    eigvals = torch.clamp(eigvals, min=1e-6)  # Ensure positivity
    cov_matrix_inv= eigvecs @ torch.diag(eigvals) @ eigvecs.T  # Reconstruct matrix
    W = torch.linalg.cholesky(cov_matrix_inv)
    return W

def compute_covariance(G_centered):
    A = G_centered.T @ G_centered
    return A

def get_real_eig(A):
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    return eigenvalues, eigenvectors


def PCA(G_centered):
    return get_real_eig(compute_covariance(G_centered))


def whiten_np(G_centered):
    cov_matrix = G_centered.T @ G_centered
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    # eigvals = np.maximum(eigvals, 1e-6)  # Ensure stability
    eigvals[eigvals < 1e-6] = 1e-6
    W_pca = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return W_pca

    # cov_matrix_inv = np.linalg.inv(cov_matrix)
    # return np.linalg.cholesky(cov_matrix_inv)


def compute_covariance_np(G_centered):
    A = G_centered.T @ G_centered
    return A

def get_real_eig_np(A):
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    return eigenvalues, eigenvectors


def PCA_np(G_centered):
    return get_real_eig_np(compute_covariance_np(G_centered))
