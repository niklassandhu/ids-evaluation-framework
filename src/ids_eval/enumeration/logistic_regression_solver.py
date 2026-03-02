from enum import Enum


class LogisticRegressionSolver(Enum):
    LIBLINEAR = "liblinear"
    NEWTON_CG = "newton-cg"
    NEWTON_CHOLESKY = "newton-cholesky"
    SAG = "sag"
    SAGA = "saga"
    LBFGS = "lbfgs"
    OTHER = "other"  # Placeholder for future methods
