"""
transforms.py
-------------
SPD-preserving covariance matrix representations and scalers.

Spec §13 – ridge-stabilized matrix-log vech representation.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import expm, logm
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lower-triangular indexing helpers for a 10×10 matrix (55 entries)
# ---------------------------------------------------------------------------

_N = 10
_LOWER_IDX = np.tril_indices(_N)   # (row_indices, col_indices)


def _vech(A: np.ndarray) -> np.ndarray:
    """Vectorize lower-triangular entries of a symmetric (N×N) matrix -> R^55."""
    return A[_LOWER_IDX]


def _ivech(v: np.ndarray) -> np.ndarray:
    """Reconstruct symmetric (N×N) matrix from its 55 lower-triangular entries."""
    n = _N
    A = np.zeros((n, n), dtype=np.float64)
    A[_LOWER_IDX] = v
    # Fill upper triangle by symmetry
    A = A + A.T - np.diag(np.diag(A))
    return A


# ---------------------------------------------------------------------------
# Primary interface
# ---------------------------------------------------------------------------

def covariance_to_log_vech(
    covariance: np.ndarray,
    ridge_epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Convert a 10×10 covariance matrix to its 55-dimensional log-matrix vech vector.

    Steps (spec §13.3):
    1. Add ridge: S_eps = S + eps * I
    2. Compute matrix log: A = logm(S_eps)
    3. Vectorize lower triangle: x = vech(A)

    Parameters
    ----------
    covariance : np.ndarray, shape (10, 10)
    ridge_epsilon : float

    Returns
    -------
    np.ndarray, shape (55,)
    """
    n = covariance.shape[0]
    S_eps = covariance + ridge_epsilon * np.eye(n)

    # Ensure symmetry before logm
    S_eps = 0.5 * (S_eps + S_eps.T)

    # Verify positive-definiteness
    eigvals = np.linalg.eigvalsh(S_eps)
    if eigvals.min() <= 0:
        raise ValueError(
            f"Ridge-stabilized covariance is not PD. "
            f"Min eigenvalue: {eigvals.min():.6e}. Increase ridge_epsilon."
        )

    A = logm(S_eps)
    # logm of a real SPD matrix is real symmetric; ensure symmetry numerically
    A = 0.5 * (A + A.T)
    return _vech(A).real.astype(np.float64)


def log_vech_to_covariance(vector: np.ndarray) -> np.ndarray:
    """
    Convert a 55-dimensional log-matrix vech vector to an SPD covariance matrix.

    Steps (spec §13.5):
    1. Reconstruct symmetric log-covariance matrix: A = ivech(v)
    2. Exponentiate: Sigma = expm(A)
    3. Symmetrize: Sigma = (Sigma + Sigma.T) / 2

    Parameters
    ----------
    vector : np.ndarray, shape (55,)

    Returns
    -------
    np.ndarray, shape (10, 10), SPD covariance matrix.
    """
    A = _ivech(vector)
    Sigma = expm(A)
    Sigma = 0.5 * (Sigma + Sigma.T)  # symmetrize numerically
    Sigma = Sigma.real.astype(np.float64)

    eigvals = np.linalg.eigvalsh(Sigma)
    if eigvals.min() <= 0:
        logger.warning(
            "Reconstructed covariance has non-positive eigenvalue: %.6e. "
            "Adding small jitter.", eigvals.min()
        )
        Sigma += (-eigvals.min() + 1e-8) * np.eye(_N)

    return Sigma


# ---------------------------------------------------------------------------
# Scaler fitting and persistence
# ---------------------------------------------------------------------------

def fit_training_scalers(
    train_condition_vectors: np.ndarray,
    train_target_vectors: np.ndarray,
    save_dir: Optional[str | Path] = None,
) -> Tuple[StandardScaler, StandardScaler]:
    """
    Fit separate StandardScalers on training conditioning and target vectors.

    Parameters
    ----------
    train_condition_vectors : (n_train, 55)
    train_target_vectors    : (n_train, 55)
    save_dir : if not None, persist scalers to PKL files.

    Returns
    -------
    (conditioning_scaler, target_scaler)
    """
    cond_scaler = StandardScaler()
    tgt_scaler = StandardScaler()

    cond_scaler.fit(train_condition_vectors)
    tgt_scaler.fit(train_target_vectors)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "conditioning_scaler.pkl", "wb") as fh:
            pickle.dump(cond_scaler, fh)
        with open(save_dir / "target_scaler.pkl", "wb") as fh:
            pickle.dump(tgt_scaler, fh)
        logger.info("Scalers saved to %s", save_dir)

    return cond_scaler, tgt_scaler


def load_scalers(
    save_dir: str | Path,
) -> Tuple[StandardScaler, StandardScaler]:
    """Load conditioning and target scalers from PKL files."""
    save_dir = Path(save_dir)
    with open(save_dir / "conditioning_scaler.pkl", "rb") as fh:
        cond_scaler = pickle.load(fh)
    with open(save_dir / "target_scaler.pkl", "rb") as fh:
        tgt_scaler = pickle.load(fh)
    return cond_scaler, tgt_scaler


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def batch_covariance_to_log_vech(
    covariances: np.ndarray,
    ridge_epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Apply covariance_to_log_vech to a batch of covariance matrices.

    Parameters
    ----------
    covariances : (n, 10, 10)
    ridge_epsilon : float

    Returns
    -------
    np.ndarray, shape (n, 55)
    """
    n = covariances.shape[0]
    out = np.empty((n, 55), dtype=np.float64)
    for i in range(n):
        out[i] = covariance_to_log_vech(covariances[i], ridge_epsilon)
    return out


def batch_log_vech_to_covariance(vectors: np.ndarray) -> np.ndarray:
    """
    Apply log_vech_to_covariance to a batch of vectors.

    Parameters
    ----------
    vectors : (n, 55)

    Returns
    -------
    np.ndarray, shape (n, 10, 10)
    """
    n = vectors.shape[0]
    out = np.empty((n, _N, _N), dtype=np.float64)
    for i in range(n):
        out[i] = log_vech_to_covariance(vectors[i])
    return out
