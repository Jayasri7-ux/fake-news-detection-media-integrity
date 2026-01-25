from sklearn.calibration import CalibratedClassifierCV, calibration_curve

class ModelCalibrator:
    def __init__(self, method='sigmoid'):
        self.method = method

    def calibrate(self, model, X_train, y_train):
        """
        Calibrates the model using the specified method (sigmoid or isotonic) with CV.
        """
        from sklearn.base import clone
        # Clone the model to get an unfitted estimator with same params
        unfitted_model = clone(model)
        
        # Use cv=3 for calibration to save time, or default
        calibrated_clf = CalibratedClassifierCV(estimator=unfitted_model, method=self.method, cv=3)
        calibrated_clf.fit(X_train, y_train)
        return calibrated_clf
