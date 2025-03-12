from __future__ import annotations

from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union


def score(clf, x, y):
    # roc_auc_score: положительный класс — это "True", т.е. y==1
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:
    def __init__(
        self,
        base_model_class=DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        early_stopping_rounds: int = None,
        
        bootstrap_type: str = 'Bernoulli',
        subsample: Union[float, int] = 1.0,
        bagging_temperature: Union[float, int] = 1.0,
    ):
        self.base_model_class = base_model_class

        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators
        self.learning_rate: float = learning_rate
        self.early_stopping_rounds = early_stopping_rounds

        self.models: list = []
        self.gammas: list = []

        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        
        self.loss_fn = lambda y, z: -(
            y * np.log(self.sigmoid(z) + 1e-12) +
            (1 - y) * np.log(1 - self.sigmoid(z) + 1e-12)
        ).mean() #заменим функцию чтобы (0, 1) учитывало на кросс-энтропию 

        self.loss_derivative = lambda y, z: (self.sigmoid(z) - y)
        
        self.bootstrap_type = bootstrap_type
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature

    def partial_fit(self, X, y):
 
        model = self.base_model_class(**self.base_model_params)
        return model.fit(X, y)

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            plot=False, show_history=False):
        
        train_predictions = np.zeros(y_train.shape[0])

        s = -self.loss_derivative(y_train, train_predictions)
        
        deadcounter = 0
        best_val_loss = float('inf')
        
        if X_val is not None and y_val is not None:
            val_predictions = np.zeros(y_val.shape[0])

        for i in range(self.n_estimators):
            
            if self.early_stopping_rounds is not None:
                if deadcounter == self.early_stopping_rounds:
                    print(f"Dead iteration: {i}")
                    break
            
            self.models.append(self.partial_fit(X_train, s))
            cur_prediction = self.models[-1].predict(X_train)

            if X_val is not None and y_val is not None:
                cur_prediction_val = self.models[-1].predict(X_val)
            
            best_gamma = self.find_optimal_gamma(y_train, train_predictions, cur_prediction)
            self.gammas.append(best_gamma)
            
            train_predictions += self.learning_rate * best_gamma * cur_prediction
            
            if X_val is not None and y_val is not None:
                val_predictions += self.learning_rate * best_gamma * cur_prediction_val
                loss_val = self.loss_fn(y_val, val_predictions)

                if loss_val < best_val_loss:
                    best_val_loss = loss_val
                    deadcounter = 0
                else:
                    deadcounter += 1
            
            s = -self.loss_derivative(y_train, train_predictions)
            
            current_roc_auc = self.score(X_train, y_train)
            current_loss = self.loss_fn(y_train, train_predictions)
            self.history["train_roc_auc"].append(current_roc_auc)
            self.history["train_loss"].append(current_loss)

            if show_history:
                print(f"iteration {i}: Loss = {current_loss:.3f}")

        if plot:
            self.plot_history(X_train, y_train)

    def predict_proba(self, X):
        full_prediction = np.zeros(X.shape[0])
        for i in range(len(self.models)):
            full_prediction += self.learning_rate * self.gammas[i] * self.models[i].predict(X)
        p1 = self.sigmoid(full_prediction)
        p0 = 1 - p1
        return np.column_stack([p0, p1])

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [
            self.loss_fn(y, old_predictions + gamma * new_predictions)
            for gamma in gammas
        ]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        return score(self, X, y)
        
    def plot_history(self, X, y):

        preds = np.zeros_like(y)
        losses = []
        
        for i in range(len(self.models)):
            preds += self.learning_rate * self.gammas[i] * self.models[i].predict(X)
            losses.append(self.loss_fn(y, preds))

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        sns.lineplot(x=range(len(losses)), y=losses, ax=ax, color='red')
        ax.set_title("Loss per iteration")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        plt.tight_layout()
