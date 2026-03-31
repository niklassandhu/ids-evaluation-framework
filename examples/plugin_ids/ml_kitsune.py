import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

# Add src/kitsune to python path to allow importing KitNET
# If you want to use Kitsune, download the source code from:
# https://github.com/ymirsky/Kitsune-py and place it according to the next line
sys.path.append(os.path.join(os.path.dirname(__file__), "../external_ids/kitsune"))

try:
    from KitNET.KitNET import KitNET
except ImportError:
    # Fallback if the path structure is different or running from a different context
    try:
        from src.kitsune.KitNET.KitNET import KitNET
    except ImportError:
        print("Error: KitNET not found. Please ensure src/kitsune/KitNET is in the python path.")
        raise

from ids_eval.interface.abstract_ids_connector import AbstractIDSConnector


class MlKitsune(AbstractIDSConnector):
    """
    Kitsune Plugin to evaluate Kitsune as proposed by:
    Y. Mirsky, T. Doitshman, Y. Elovici, und A. Shabtai,
    „Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection“,
    arXiv.org. Zugegriffen: 26. November 2025. [Online]. https://arxiv.org/abs/1802.09089v2
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.threshold = 0.0
        self.dist_mean = 0.0
        self.dist_std = 1.0
        self.params = {}

    def _ids_deploy(self, params: dict[str, Any]) -> None:
        """
        Initialize the Kitsune model.
        Since KitNET requires the number of features (n) which might not be known until prepare,
        we store the params and initialize the model in _ids_prepare if n is not in params.
        """
        self.params = params
        self.logger.info(f"Deploying Kitsune with params: {params}")

        # If n_features is known, we can initialize here, but it's safer to wait for data in prepare
        # unless explicitly provided.
        if "n_features" in params:
            self._init_model(params["n_features"])

    def _init_model(self, n_features: int):
        max_ae = self.params.get("max_autoencoder_size", 10)
        fm_grace = self.params.get("FM_grace_period", None)  # Default None means equal to AD_grace
        ad_grace = self.params.get("AD_grace_period", 10000)
        lr = self.params.get("learning_rate", 0.1)
        hidden_ratio = self.params.get("hidden_ratio", 0.75)

        self.model = KitNET(
            n=n_features,
            max_autoencoder_size=max_ae,
            FM_grace_period=fm_grace,
            AD_grace_period=ad_grace,
            learning_rate=lr,
            hidden_ratio=hidden_ratio,
        )
        self.logger.info(
            f"Initialized KitNET with n={n_features}, max_ae={max_ae}, FM_grace={fm_grace}, AD_grace={ad_grace}"
        )

    def _ids_prepare(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        # Preparation is according to Kitsune provided example.py in source code
        n_features = x_train.shape[1]
        n_samples = x_train.shape[0]

        if self.model is None:
            # Check if we need to adjust grace periods based on data size
            # If defaults are too large for the data, scale them down
            ad_grace = self.params.get("AD_grace_period", 10000)
            fm_grace = self.params.get("FM_grace_period", ad_grace)

            total_grace = fm_grace + ad_grace
            if total_grace >= n_samples:
                self.logger.warning(
                    f"Training data size ({n_samples}) is smaller than or equal to total grace period ({total_grace}). "
                    "Adjusting grace periods to fit data."
                )
                # Strategy: Use 25% for FM, 50% for AD, leaving 25% for threshold estimation
                fm_grace = int(n_samples * 0.25)
                ad_grace = int(n_samples * 0.50)
                # Update params so _init_model uses them
                self.params["FM_grace_period"] = fm_grace
                self.params["AD_grace_period"] = ad_grace

            self._init_model(n_features)

        self.logger.info(f"Training Kitsune on {n_samples} samples...")

        rmses = []
        for i in range(n_samples):
            x = x_train.iloc[i].values
            rmse = self.model.process(x)

            # process returns 0.0 during grace periods (training)
            # and RMSE > 0.0 after grace periods (execution)
            if rmse > 0.0:
                rmses.append(rmse)

        if not rmses:
            self.logger.error(
                "No RMSEs collected during training. Grace periods covered all data. "
                "Please reduce FM_grace_period and AD_grace_period or provide more training data."
            )
            raise ValueError(
                "Cannot train Kitsune: No RMSEs collected. " "Grace periods are too large for the training data size."
            )

        # Estimate distribution parameters
        # Following example.py: fit log-normal distribution
        rmses = np.array(rmses)
        # Add small epsilon to avoid log(0) for edge cases with perfect reconstruction
        log_rmses = np.log(rmses + 1e-10)
        self.dist_mean = np.mean(log_rmses)
        self.dist_std = np.std(log_rmses)

        # Handle edge case where all RMSEs are identical (std = 0)
        if self.dist_std < 1e-10:
            self.logger.warning("All RMSEs are nearly identical. Using default std=1.0")
            self.dist_std = 1.0

        self.logger.info(f"Training complete. Collected {len(rmses)} RMSEs.")
        self.logger.info(f"Log-Normal params: mean={self.dist_mean:.4f}, std={self.dist_std:.4f}")

        # Calculate threshold based on configured method
        threshold_method = self.params.get("threshold_method", "sigma")

        if threshold_method == "sigma":
            self._calculate_threshold_sigma(rmses)
        elif threshold_method == "percentile":
            self._calculate_threshold_percentile(rmses)
        else:
            self.logger.warning(f"Unknown threshold_method '{threshold_method}', using 'sigma'")
            self._calculate_threshold_sigma(rmses)

    def _calculate_threshold_sigma(self) -> None:
        """Calculate threshold using sigma coefficient on log-normal distribution."""
        sigma_coeff = self.params.get("threshold_sigma", 3.0)
        log_threshold = self.dist_mean + sigma_coeff * self.dist_std
        self.threshold = np.exp(log_threshold)
        self.logger.info(f"Threshold (sigma method): {self.threshold:.6f} (sigma={sigma_coeff})")

    def _calculate_threshold_percentile(self, rmses: np.ndarray) -> None:
        """Calculate threshold using a fixed percentile of training RMSEs."""
        percentile = self.params.get("threshold_percentile", 95.0)
        self.threshold = np.percentile(rmses, percentile)
        self.logger.info(f"Threshold (percentile method): {self.threshold:.6f} ({percentile}th percentile)")

    def _ids_detect(self, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n_samples = x_test.shape[0]
        y_pred = np.zeros(n_samples, dtype=int)
        y_proba = np.zeros(n_samples, dtype=float)

        self.logger.info(f"Detecting on {n_samples} samples...")

        for i in range(n_samples):
            x = x_test.iloc[i].values
            # Use execute() to ensure no training happens on test data
            rmse = self.model.execute(x)

            # Calculate probability (confidence score)
            # Using log-normal CDF
            # We want probability of being anomalous.
            # If rmse is high, probability should be high.
            # norm.cdf(log(rmse), loc=mean, scale=std) gives probability that a value is <= log(rmse).
            # So if log(rmse) is very high (far right), cdf is close to 1.
            # This works as an anomaly score (0 to 1).
            if rmse <= 0:
                prob = 0.0
            else:
                log_rmse = np.log(rmse)
                prob = norm.cdf(log_rmse, loc=self.dist_mean, scale=self.dist_std)

            y_proba[i] = prob

            # Binary prediction
            if rmse > self.threshold:
                y_pred[i] = 1
            else:
                y_pred[i] = 0

        return y_pred, y_proba

    def _ids_save(self, path: Path) -> None:
        model_file = path / "kitnet_model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(self.model, f)

        aux_data = {
            "threshold": self.threshold,
            "dist_mean": self.dist_mean,
            "dist_std": self.dist_std,
            "params": self.params,
        }
        aux_file = path / "aux_params.pkl"
        with open(aux_file, "wb") as f:
            pickle.dump(aux_data, f)

    def _ids_load(self, path: Path) -> bool:
        model_file = path / "kitnet_model.pkl"
        aux_file = path / "aux_params.pkl"

        if not model_file.exists() or not aux_file.exists():
            return False

        with open(model_file, "rb") as f:
            self.model = pickle.load(f)

        with open(aux_file, "rb") as f:
            aux_data = pickle.load(f)

        self.threshold = aux_data["threshold"]
        self.dist_mean = aux_data["dist_mean"]
        self.dist_std = aux_data["dist_std"]
        self.params = aux_data["params"]

        return True
