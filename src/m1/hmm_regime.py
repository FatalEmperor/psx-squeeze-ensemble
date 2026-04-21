"""
Layer 4: Hidden Markov Model regime filter.

Fits a 2-state Gaussian HMM on log-returns and labels each bar as
bull (trade) or bear/choppy (skip).
"""

import numpy as np
from hmmlearn.hmm import GaussianHMM

from config import HMM_STATES


def fit_hmm(log_returns: np.ndarray, n_states: int = HMM_STATES) -> GaussianHMM:
    """
    Fit a Gaussian HMM to the full log-return series.

    Parameters
    ----------
    log_returns : 1-D array of log returns
    n_states    : number of hidden states (default 2)

    Returns
    -------
    Fitted GaussianHMM model
    """
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )
    model.fit(log_returns.reshape(-1, 1))
    return model


def hmm_bull_states(model: GaussianHMM, log_returns: np.ndarray) -> np.ndarray:
    """
    Decode hidden states and return a boolean mask where True = bull regime.

    Bull state = the state with the higher mean log-return.

    Parameters
    ----------
    model       : fitted GaussianHMM
    log_returns : 1-D array of log returns (same series used to fit)

    Returns
    -------
    Boolean ndarray, length == len(log_returns)
    """
    states     = model.predict(log_returns.reshape(-1, 1))
    bull_state = int(np.argmax(model.means_.flatten()))
    return states == bull_state
