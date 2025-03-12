from __future__ import annotations

from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:
    def __init__(
        self,
        base_model_class=DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        early_stopping_rounds: int = None,
        bootstrap_type: Optional[str] = 'Bernoulli',
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

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -(
            y * np.log(self.sigmoid(z) + 1e-12)
            + (1 - y) * np.log(1 - self.sigmoid(z) + 1e-12)
        ).mean() #поменял, чтобы не было траблов с y (0, 1) то есть работаем с классической кросс энтропией 

        self.loss_derivative = lambda y, z: (self.sigmoid(z) - y) #соответственно производная тоже не поменялась 

        self.bootstrap_type = bootstrap_type
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature

    def partial_fit(self, X, y, sample_weight=None):
        model = self.base_model_class(**self.base_model_params)
        return model.fit(X, y, sample_weight=sample_weight)

    def _get_bootstrap(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        s: np.ndarray
    ):
        n = X_train.shape[0]

        if self.bootstrap_type == 'Bernoulli':
            if isinstance(self.subsample, float):
                mask = np.random.rand(n) < self.subsample
                X_boot = X_train[mask]
                y_boot = y_train[mask]
                s_boot = s[mask]
                weights_boot = None 
                
            elif isinstance(self.subsample, int):
                idx = np.random.choice(n, size=self.subsample, replace=False)
                X_boot = X_train[idx]
                y_boot = y_train[idx]
                s_boot = s[idx]
                weights_boot = None
            else:
                raise ValueError()
        
        elif self.bootstrap_type == 'Bayesian':
            U = np.random.rand(n)
            w = (-np.log(U)) ** self.bagging_temperature
            X_boot = X_train
            y_boot = y_train
            s_boot = s
            weights_boot = w       
        else:
            X_boot = X_train
            y_boot = y_train
            s_boot = s
            weights_boot = None

        return X_boot, y_boot, s_boot, weights_boot

    def fit(self, 
            X_train, 
            y_train,
            X_val=None, 
            y_val=None,
            plot=False, 
            show_history=False):

        train_predictions = np.zeros_like(y_train, dtype=float)
        s = -self.loss_derivative(y_train, train_predictions)
        
        deadcounter = 0
        best_val_loss = float('inf')
        
        if X_val is not None and y_val is not None:
            val_predictions = np.zeros_like(y_val, dtype=float)

        for i in range(self.n_estimators):

            if self.early_stopping_rounds is not None:
                if deadcounter == self.early_stopping_rounds:
                    print(f"Dead Iteration {i}")
                    break

            X_boot, y_boot, s_boot, weights_boot = self._get_bootstrap(X_train, y_train, s)

            model = self.partial_fit(X_boot, s_boot, sample_weight=weights_boot)
            self.models.append(model)

            cur_prediction = model.predict(X_train)

            if X_val is not None and y_val is not None:
                cur_prediction_val = model.predict(X_val)

            best_gamma = self.find_optimal_gamma(y_train, train_predictions, cur_prediction)
            self.gammas.append(best_gamma)

            train_predictions += self.learning_rate * best_gamma * cur_prediction

            if X_val is not None and y_val is not None:
                val_predictions += self.learning_rate * best_gamma * cur_prediction_val
                loss_val = self.loss_fn(y_val, val_predictions)
                if loss_val <= best_val_loss:
                    best_val_loss = loss_val
                    deadcounter = 0
                else:
                    deadcounter += 1

            s = -self.loss_derivative(y_train, train_predictions)

            current_loss = self.loss_fn(y_train, train_predictions)
            current_auc = self.score(X_train, y_train)
            self.history["train_loss"].append(current_loss)
            self.history["train_roc_auc"].append(current_auc)

            if show_history:
                print(f"Iteration {i}: loss={current_loss:.5f}")

        if plot:
            self.plot_history(X_train, y_train)

    def predict_proba(self, X):

        full_prediction = np.zeros(X.shape[0], dtype=float)
        for model, gamma in zip(self.models, self.gammas):
            full_prediction += self.learning_rate * gamma * model.predict(X)
        p1 = self.sigmoid(full_prediction)
        p0 = 1 - p1
        return np.column_stack([p0, p1])

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:

        gammas = np.linspace(0, 1, 100)
        losses = [
            self.loss_fn(y, old_predictions + gamma * new_predictions)
            for gamma in gammas
        ]
        return gammas[np.argmin(losses)]

    def score(self, X, y):

        return score(self, X, y)

    def plot_history(self, X, y, title='Loss per iteration', color='red'):

        partial_pred = np.zeros_like(y, dtype=float)
        losses = []
        for i in range(len(self.models)):
            partial_pred += self.learning_rate * self.gammas[i] * self.models[i].predict(X)
            losses.append(self.loss_fn(y, partial_pred))

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.lineplot(x=range(len(losses)), y=losses, ax=ax, color=color)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        plt.tight_layout()
        plt.show()
