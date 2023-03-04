import os

import numpy as np
from matplotlib import pyplot as plt

from utils import *


class LogisticRegression(object):
    """
    Logistic regression.

    Shape D means the dimension of the feature.
    Shape N means the number of the training examples.

    Attributes:
        weights: The weight vector of shape (D, 1).
        bias: The bias term.
    """

    def __init__(self) -> None:
        """
        Initialize the parameters of the logistic regression by setting parameters
        weights and bias to None

        Args:
            None.

        Returns:
            None.
        """
        self.weights = None
        self.bias = None
    

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid function.

        Args:
            z: The input of shape (N,).

        Returns:
            sigmoid: The sigmoid output of shape (N,).
        """

        # >> YOUR CODE HERE
        sigmoid = ...
        # << END OF YOUR CODE

        # The following part is for avoiding the value to be 0 or 1. DO NOT MODIFY.
        sigmoid[sigmoid > 0.99] = 0.99
        sigmoid[sigmoid < 0.01] = 0.01
        return sigmoid

    def gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to the weights by first
        computing the probabilities of the labels for training data and then
        computing the gradient with respect to the weights.

        Args:
            X_train: The training features of shape (N, D).
            y_train: The training labels of shape (N,).

        Returns:
            grad_w: The gradient of the loss with respect to the weights of
                shape (D, 1).
            grad_b: The gradient of the loss with respect to the bias of
                shape (1,).
        """

        grad_w, grad_b = None, None

        # >> YOUR CODE HERE
        ...
        # << END OF YOUR CODE

        return grad_w, grad_b

    def logistic_loss(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Compute the loss of the logistic regression by first computing the
        probabilities of the labels for training data and then computing the
        loss.

        Args:
            X_train: The training features of shape (N, D).
            y_train: The training labels of shape (N,).

        Returns:
            logistic_loss: The logistic loss.
        """

        logistic_loss = None
        
        # >> YOUR CODE HERE
        ...
        # << END OF YOUR CODE

        return logistic_loss

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the probability of the labels given the features, using the
        weights and bias of the logistic regression and the sigmoid function.

        Args:
            X: The features of shape (N, D).

        Returns:
            y_pred_proba: The probabilities of the labels in numpy array with shape (N,).
        """
        
        y_pred_proba = None

        # >> YOUR CODE HERE
        ...
        # << END OF YOUR CODE

        return y_pred_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions given new inputs by first computing the probabilities
        and then rounding them to the closest integer. 

        Args:
            X: The features of shape (N, D).

        Returns:
            y_pred: Predicted labels of shape (N,).
        """

        y_pred = None

        # >> YOUR CODE HERE
        ...
        # << END OF YOUR CODE

        return y_pred

    def train_one_epoch(self, X_train: np.ndarray, y_train: np.ndarray,
                        learning_rate: float = 0.001) -> None:
        """
        Train the logistic regression for one epoch. First compute the
        the gradients with respect to the weights and bias, and then update the
        weights and bias.

        Args:
            X_train: The training features of shape (N, D).
            y_train: The training labels of shape (N,).
            learning_rate: The learning rate, default is 0.001.

        Returns:
            grad_w: The gradient of the loss with respect to the weights of
                shape (D, 1).
            grad_b: The gradient of the loss with respect to the bias of
                shape (1,).
        """

        grad_w, grad_b = None, None

        # >> YOUR CODE HERE
        ...
        # << END OF YOUR CODE

        return grad_w, grad_b

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_valid: np.ndarray,
              y_valid: np.ndarray,
              max_epochs: int = 50000,
              lr: float = 0.001,
              tol: float = 1e-2) -> None:
        """
        Train the logistic regression using gradient descent. First initialize
        the weights and bias, and then train the model for max_epochs iterations
        by calling train_one_epoch() If the absolute value of the gradient of
        weights is less than tol, stop training. You may use self.logistic_loss()
        and accuracy() (in utils.py) to compute the loss and accuracy of the model and 
        print them out during training.


        Args:
            X_train: The training features of shape (N, D).
            y_train: The training labels of shape (N,).
            max_epochs: The maximum number of epochs, default is 50000.
            lr: The learning rate, default is 0.001.
            tol: The tolerance for early stopping, default is 1e-2.

        Returns:
            None.
        """

        self.weights = np.random.randn(X_train.shape[1])
        self.bias = np.random.randn()

        for epoch in range(max_epochs):

            # >> YOUR CODE HERE
            ...
            train_loss = ...
            valid_loss = ...
            train_acc = ...
            valid_acc = ...
            # << END OF YOUR CODE

            if epoch % 100 == 0:
                print(
                    f'Epoch {epoch}: train loss = {train_loss:.8f}, valid loss = {valid_loss:.8f}, train acc = {train_acc:.8f}, valid acc = {valid_acc:.8f}')

        print(
            f'Final: train loss = {train_loss:.8f}, valid loss = {valid_loss:.8f}, train acc = {train_acc:.8f}, valid acc = {valid_acc:.8f}')


def plot_logistic_regression_curve(X_train: np.ndarray,
                                   y_train: np.ndarray,
                                   X_valid: np.ndarray,
                                   y_valid: np.ndarray,
                                   max_epochs: int = 50000,
                                   lr: float = 0.001,
                                   ) -> None:
    """
    Plot loss and accuracy curves for the logistic regression classifier.

    Args:
        logistic_regression_classifier: The logistic regression classifier.
        X_train: The training features of shape (N, D).
        y_train: The training labels of shape (N,).
        X_valid: The validation features of shape (N, D).
        y_valid: The validation labels of shape (N,).
        max_epochs: The maximum number of epochs.
        lr: The learning rate.
        tol: The tolerance for early stopping.

    Returns:
        None
    """
    lr_classifier = LogisticRegression()
    lr_classifier.weights = np.random.randn(X_train.shape[1])
    lr_classifier.bias = np.random.randn()

    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []

    for _ in range(max_epochs):

        ## >>> YOUR CODE HERE
        ...
        ## <<< END OF YOUR CODE

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(valid_acc, label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(os.path.dirname(
        __file__), "learning_curve_lr.png"))

"""
-------------------------------------------------------------------------------------------
THE CODE BELOW IS FOR EVALUATION. PLEASE DO NOT CHANGE!
-------------------------------------------------------------------------------------------
"""

def evaluate_lr():
    X, y = load_data(os.path.join(
        os.path.dirname(__file__), 'dataset/dating_train.csv'))
    X_train, X_valid, y_train, y_valid = my_train_test_split(
        X, y, 0.2, random_state=42)

    print('\n\n-------------Fitting Logistic Regression-------------\n')
    lr = LogisticRegression()
    lr.train(X_train, y_train, X_valid,
             y_valid, max_epochs=12000, lr=0.001)

    print('\n\n-------------Logistic Regression Performace-------------\n')
    evaluate(y_train,
             lr.predict(X_train),
             y_valid,
             lr.predict(X_valid))

    print('\n\n-------------Plotting learning curves-------------\n')

    print('Plotting Logistic Regression learning curves...')

    plot_logistic_regression_curve(
        LogisticRegression(), X_train, y_train, X_valid, y_valid, max_epochs=12000, lr=0.001)

if __name__ == '__main__':

    os.system('cls' if os.name == 'nt' else 'clear')

    print('\n-------------Logistic Regression-------------')
    evaluate_lr()

    print('\nDone!')
