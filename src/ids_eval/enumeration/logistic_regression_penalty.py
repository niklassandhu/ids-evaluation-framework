from enum import Enum

from ids_eval.enumeration.logistic_regression_solver import LogisticRegressionSolver


class LogisticRegressionPenalty(Enum):
    L1 = "l1"
    L2 = "l2"
    ELASTICNET = "elasticnet"
    NONE = "none"
    OTHER = "other"  # Placeholder for future methods

    def validate(self, solver: LogisticRegressionSolver) -> bool:
        if self == LogisticRegressionPenalty.L1:
            return solver in {LogisticRegressionSolver.LIBLINEAR, LogisticRegressionSolver.SAGA}
        elif self == LogisticRegressionPenalty.L2:
            return solver in {
                LogisticRegressionSolver.LBFGS,
                LogisticRegressionSolver.NEWTON_CG,
                LogisticRegressionSolver.NEWTON_CHOLESKY,
                LogisticRegressionSolver.SAG,
                LogisticRegressionSolver.SAGA,
                LogisticRegressionSolver.LIBLINEAR,
            }
        elif self == LogisticRegressionPenalty.ELASTICNET:
            return solver == LogisticRegressionSolver.SAGA
        elif self == LogisticRegressionPenalty.NONE:
            return solver in {
                LogisticRegressionSolver.LBFGS,
                LogisticRegressionSolver.NEWTON_CG,
                LogisticRegressionSolver.NEWTON_CHOLESKY,
                LogisticRegressionSolver.SAG,
                LogisticRegressionSolver.SAGA,
            }
        elif self == LogisticRegressionPenalty.OTHER:
            return True  # Assume 'other' is always valid
        return False
