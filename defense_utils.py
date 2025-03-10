import torch


def whiten(G_centered):
    cov_matrix = G_centered.T @ G_centered
    cov_matrix_inv = torch.linalg.inv(cov_matrix)
    W = torch.linalg.cholesky(cov_matrix_inv)
    return W

def compute_covariance(G_centered):
    A = G_centered.T @ G_centered
    return A

def get_real_eig(A):
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    # if torch.all(torch.abs(torch.imag(eigenvalues)) < 1e-10):
    #     eigenvalues = torch.real(eigenvalues)
    #     eigenvectors = torch.real(eigenvectors)
    # else:
    #     raise ValueError
    return eigenvalues, eigenvectors


def PCA(G_centered):
    return get_real_eig(compute_covariance(G_centered))