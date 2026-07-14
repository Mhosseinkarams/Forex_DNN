import numpy as np
from abc import ABC, abstractmethod


class BaseCalibrator(ABC):
    """
    Abstract Base Class for model prediction probability calibration.
    """
    @abstractmethod
    def calibrate(self, probability: float) -> float:
        """
        Calibrate a raw probability score.
        """
        pass


class IdentityCalibrator(BaseCalibrator):
    """
    Passes raw probabilities through without modification (identity calibration).
    """
    def calibrate(self, probability: float) -> float:
        return float(np.clip(probability, 0.0, 1.0))


class PlattCalibrator(BaseCalibrator):
    """
    Standard Platt scaling logic helper.
    Assuming pre-trained or standard logistic mapping parameters:
    P(y=1|x) = 1 / (1 + exp(A * f(x) + B))
    Where f(x) is mapped from the raw probability via logit.
    For this decoupled inference-side class, we can configure A and B.
    """
    def __init__(self, A: float = -1.0, B: float = 0.0):
        self.A = A
        self.B = B

    def calibrate(self, probability: float) -> float:
        # Convert raw probability to dummy logit score
        eps = 1e-15
        probability = np.clip(probability, eps, 1.0 - eps)
        f_x = np.log(probability / (1.0 - probability))

        calibrated = 1.0 / (1.0 + np.exp(self.A * f_x + self.B))
        return float(np.clip(calibrated, 0.0, 1.0))


class IsotonicCalibrator(BaseCalibrator):
    """
    Piecewise linear isotonic calibrator.
    Maps probability inputs based on configured step coordinates.
    """
    def __init__(self, thresholds: list = None, targets: list = None):
        # Default maps 0.0->0.0, 0.5->0.4, 0.8->0.85, 1.0->1.0
        self.thresholds = thresholds or [0.0, 0.5, 0.8, 1.0]
        self.targets = targets or [0.0, 0.4, 0.85, 1.0]

    def calibrate(self, probability: float) -> float:
        # Interpolate the raw probability onto the targets mapping piecewise-linearly
        val = float(np.interp(probability, self.thresholds, self.targets))
        return float(np.clip(val, 0.0, 1.0))
