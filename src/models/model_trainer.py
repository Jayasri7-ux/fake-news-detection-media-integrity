from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    def train(self, model, param_grid):
        """
        Trains a model using GridSearchCV for hyperparameter tuning.
        
        Args:
            model: The base estimator.
            param_grid (dict): Dictionary with parameters names as keys and lists of parameter settings to try as values.
            
        Returns:
            The best estimator found by GridSearchCV.
        """
        # Using 3-fold CV to save time during this example validation, can be increased.
        grid = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1) 
        grid.fit(self.X_train, self.y_train)

        print(f"Best parameters for {type(model).__name__}: {grid.best_params_}")
        print(f"Best score: {grid.best_score_}")

        return grid.best_estimator_

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
