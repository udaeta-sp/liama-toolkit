"""Multivariate analysis: PCA, PLS-DA, Random Forest.

Each analysis returns results that can be displayed and exported,
plus parameters needed for projecting new samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class PCAResult:
    scores: np.ndarray                    # (n_samples, n_components)
    loadings: np.ndarray                  # (n_components, n_features)
    explained_variance_ratio: np.ndarray  # (n_components,)
    pca_model: PCA = field(repr=False)
    sample_names: list[str] = field(default_factory=list)
    labels: np.ndarray | None = None
    wavenumbers: np.ndarray | None = None
    # Preprocessing params for projection
    fit_params: dict = field(default_factory=dict)
    pipeline_steps: list[dict] = field(default_factory=list)

    def project(self, X_new: np.ndarray) -> np.ndarray:
        """Project new samples using the fitted PCA.

        X_new must already be preprocessed with the same pipeline.
        If scaling was used, apply it with saved params:
            X_new_scaled = (X_new - fit_params['scale_means']) / fit_params['scale_stds']
        Then: scores_new = pca_model.transform(X_new_scaled)
        """
        if "scale_means" in self.fit_params:
            means = self.fit_params["scale_means"]
            stds = self.fit_params["scale_stds"]
            X_new = (X_new - means) / stds
        return self.pca_model.transform(X_new)

    def loading_as_spectrum(self, component: int) -> np.ndarray:
        """Get a specific PC loading vector (to plot as a spectrum)."""
        return self.loadings[component]


@dataclass
class PLSDAResult:
    scores: np.ndarray
    y_pred_train: np.ndarray
    y_pred_test: np.ndarray
    y_true_test: np.ndarray
    classes: np.ndarray
    confusion: np.ndarray
    report: str
    model: PLSRegression = field(repr=False)
    sample_names: list[str] = field(default_factory=list)
    labels: np.ndarray | None = None
    train_mask: np.ndarray | None = None
    test_mask: np.ndarray | None = None


@dataclass
class RFResult:
    importances: np.ndarray
    wavenumbers: np.ndarray
    y_pred_test: np.ndarray
    y_true_test: np.ndarray
    classes: np.ndarray
    confusion: np.ndarray
    report: str
    model: RandomForestClassifier = field(repr=False)
    sample_names: list[str] = field(default_factory=list)
    train_mask: np.ndarray | None = None
    test_mask: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def run_pca(
    X: np.ndarray,
    n_components: int | None = None,
    sample_names: list[str] | None = None,
    labels: np.ndarray | None = None,
    wavenumbers: np.ndarray | None = None,
    fit_params: dict | None = None,
    pipeline_steps: list[dict] | None = None,
) -> PCAResult:
    """Run PCA on preprocessed data matrix X (n_samples × n_features).

    Operation:
        pca = sklearn.decomposition.PCA(n_components=n)
        scores = pca.fit_transform(X)
        loadings = pca.components_
    """
    n_max = min(X.shape[0], X.shape[1])
    if n_components is None:
        n_components = min(10, n_max)
    else:
        n_components = min(n_components, n_max)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    return PCAResult(
        scores=scores,
        loadings=pca.components_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        pca_model=pca,
        sample_names=sample_names or [],
        labels=labels,
        wavenumbers=wavenumbers,
        fit_params=fit_params or {},
        pipeline_steps=pipeline_steps or [],
    )


def run_plsda(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
    sample_names: list[str] | None = None,
) -> PLSDAResult:
    """Run PLS-DA classification.

    Operation:
        encoder = OneHotEncoder(sparse_output=False)
        Y_onehot = encoder.fit_transform(y.reshape(-1, 1))
        pls = PLSRegression(n_components=min(10, n_samples-1, n_features))
        pls.fit(X_train, Y_train)
        y_pred = classes[argmax(pls.predict(X_test), axis=1)]
    """
    classes = np.unique(y)

    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    encoder = OneHotEncoder(sparse_output=False)
    Y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    n_components = min(10, X_train.shape[0] - 1, X_train.shape[1])
    n_components = max(1, n_components)
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, Y_train)

    # Scores for all samples
    scores = pls.transform(X)

    # Predictions
    Y_pred_train = pls.predict(X_train)
    y_pred_train = classes[np.argmax(Y_pred_train, axis=1)]
    Y_pred_test = pls.predict(X_test)
    y_pred_test = classes[np.argmax(Y_pred_test, axis=1)]

    cm = confusion_matrix(y_test, y_pred_test, labels=classes)
    report = classification_report(y_test, y_pred_test, labels=classes, zero_division=0)

    return PLSDAResult(
        scores=scores,
        y_pred_train=y_pred_train,
        y_pred_test=y_pred_test,
        y_true_test=y_test,
        classes=classes,
        confusion=cm,
        report=report,
        model=pls,
        sample_names=sample_names or [],
        labels=y,
        train_mask=idx_train,
        test_mask=idx_test,
    )


def run_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    wavenumbers: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
    n_estimators: int = 500,
    sample_names: list[str] | None = None,
) -> RFResult:
    """Run Random Forest classification.

    Operation:
        rf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_
    """
    classes = np.unique(y)

    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    rf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred_test = rf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_test, labels=classes)
    report = classification_report(y_test, y_pred_test, labels=classes, zero_division=0)

    return RFResult(
        importances=rf.feature_importances_,
        wavenumbers=wavenumbers,
        y_pred_test=y_pred_test,
        y_true_test=y_test,
        classes=classes,
        confusion=cm,
        report=report,
        model=rf,
        sample_names=sample_names or [],
        train_mask=idx_train,
        test_mask=idx_test,
    )
