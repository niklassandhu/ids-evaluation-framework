import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from ids_eval.interface.abstract_ids_connector import AbstractIDSConnector


class MultiArmedBanditThompsonSampling:
    """
    Copied from the Apollon repo, reference implementation: `MAB/mab.ipynb`
    (class `MultiArmedBanditThompsonSampling`):https://github.com/antoniopaya22/apollon
    """

    def __init__(self, n_arms, n_clusters):
        self.n_arms = n_arms
        self.n_clusters = n_clusters
        mlp = MLPClassifier()
        self.arms = [RandomForestClassifier(), DecisionTreeClassifier(),
                     mlp, LogisticRegression(max_iter=1000), GaussianNB()]
        self.cluster_centers = None
        self.cluster_assignments = None
        self.reward_sums = {}
        for cluster in range(n_clusters):
            self.reward_sums[cluster] = np.zeros(n_arms)
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)

    def train(self, X_train, y_train):
        kmeans = KMeans(n_clusters=self.n_clusters)
        self.cluster_assignments = kmeans.fit_predict(X_train)
        self.cluster_centers = kmeans.cluster_centers_
        # Print the number of samples in each cluster

        for i in range(self.n_clusters):
            #print('Cluster {}: {}'.format(
            #    i, np.sum(self.cluster_assignments == i)))
            cluster_mask = self.cluster_assignments == i
            cluster_X_train = X_train[cluster_mask]
            cluster_y_train = y_train[cluster_mask]
            for arm in range(self.n_arms):
                #print('Training arm {} on cluster {}'.format(arm, i))
                arm_mask = cluster_y_train == arm
                arm_X_train = cluster_X_train[arm_mask]
                arm_y_train = cluster_y_train[arm_mask]
                if len(arm_X_train) > 0 and len(np.unique(arm_y_train)) > 1:
                    self.arms[arm].fit(arm_X_train, arm_y_train)
                else:
                    self.arms[arm].fit(X_train, y_train)

        # Set the arms rewards for each cluster
        for i in range(self.n_clusters):
            cluster_mask = self.cluster_assignments == i
            cluster_X_test = X_train[cluster_mask]
            cluster_y_test = y_train[cluster_mask]
            for arm in range(self.n_arms):
                #print('Setting reward_sums arm {} on cluster {}'.format(arm, i))
                arm_mask = cluster_y_test == arm
                arm_X_test = cluster_X_test[arm_mask]
                arm_y_test = cluster_y_test[arm_mask]
                if len(arm_X_test) > 0:
                    arm_y_pred = self.arms[arm].predict(arm_X_test)
                    self.reward_sums[i][arm] = np.mean(
                        arm_y_pred == arm_y_test)

    def select_arm(self, cluster):
        # Select the arm with the highest reward
        theta = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            theta[arm] = np.random.beta(self.alpha[arm] + self.reward_sums[cluster]
                                        [arm], self.beta[arm] + 1 - self.reward_sums[cluster][arm])
        return np.argmax(theta)

    def predict(self, X_test):
        """Route each sample to its nearest cluster's sampled arm and predict."""
        # Select the arm for each sample
        arms = np.zeros(len(X_test))
        for i in range(len(X_test)):
            cluster = np.argmin(np.linalg.norm(
                self.cluster_centers - X_test[i], axis=1))
            arms[i] = self.select_arm(cluster)
        # Predict using the selected arm
        y_pred = np.zeros(len(X_test))
        for arm in range(self.n_arms):
            arm_mask = arms == arm
            arm_X_test = X_test[arm_mask]
            if len(arm_X_test) > 0:
                y_pred[arm_mask] = self.arms[arm].predict(arm_X_test)
        return y_pred, arms


class ApollonIds(AbstractIDSConnector):
    """
    Wrapper for MultiArmedBanditThompsonSampling above.
    Based on: A. Paya, S. Arroni, V. Garcia-Diaz, A. Gonzalez-Diaz,
    "Apollon: A robust defense system against Adversarial Machine Learning
    attacks in Intrusion Detection Systems", Computers & Security, 2023.
    doi: 10.1016/j.cose.2023.103469.
    Reference implementation: https://github.com/antoniopaya22/apollon
    """

    def __init__(self):
        super().__init__()
        self.mab: MultiArmedBanditThompsonSampling | None = None

    def _ids_deploy(self, params: dict[str, Any]) -> None:
        """Build the MAB ensemble from params (n_arms, n_clusters)."""
        n_arms      = params["n_arms"]
        n_clusters  = params["n_clusters"]
        self.mab    = MultiArmedBanditThompsonSampling(n_arms=n_arms, n_clusters=n_clusters)
        self.logger.info(f"Deployed Apollon MAB with n_arms={n_arms}, n_clusters={n_clusters}")

    def _ids_prepare(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the MAB (pandas -> numpy)."""
        X = x_train.to_numpy()  #original expects np, not pd
        y = y_train.to_numpy()  # same
        self.mab.train(X, y)

    def _ids_detect(self, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict labels and reassemble per-sample class probabilities."""
        X = x_test.to_numpy()  # pd -> np bc of original code uses np
        y_pred, arms = self.mab.predict(X)

        #we need probabilities not only discreet label
        n_samples = len(X)
        y_proba: np.ndarray | None = None
        for arm_idx, arm_model in enumerate(self.mab.arms):
            arm_mask = arms == arm_idx
            if not arm_mask.any():
                continue
            arm_proba = arm_model.predict_proba(
                X[arm_mask]
            )
            if y_proba is None:
                y_proba = np.zeros(
                    (n_samples, arm_proba.shape[1])
                )
            y_proba[arm_mask] = arm_proba

        return y_pred, y_proba

    def _ids_save(self, path: Path) -> None:
        """Pickle the MAB's internal state to ``path``."""
        with open(path / "mab_state.pkl", "wb") as f:
            pickle.dump(self.mab.__dict__, f)

    def _ids_load(self, path: Path) -> bool:
        """Restore the MAB from a pickled state; return False if none exists."""
        state_file = path / "mab_state.pkl"
        if not state_file.exists():
            return False
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        self.mab = MultiArmedBanditThompsonSampling(
            n_arms=state["n_arms"], n_clusters=state["n_clusters"]
        )
        self.mab.__dict__.update(state)
        return True
