import matplotlib.pyplot as plt
import numpy as np
from utils import *

class NaiveBayesClassifier:

	def __init__(self, alpha):
		"""
		Input:
			alpha - Integer parameter for Laplacian smoothing, typically set to 1
		"""
		self.alpha = alpha

	def compute_class_probability(self, y_train):
		"""
		Input: 
			y_train: Numpy array of class labels corresponding to the train data
		Return: 
			class_probabilities: dictionary where each key is a class label and 
				corresponding value is the probability of that label in y_train
		"""
		
		class_probabilities = {}

		## >>> YOUR CODE HERE >>>
		...
		## <<< END OF YOUR CODE <<<

		return class_probabilities
		
	def compute_feature_probability(self, Xj_train, y_train):
		"""
		Input:
			Xj_train: a 1D array of strings with the values of a given feature 
				X_j for all data points
			y_train: a 1D array of strings with the class labels of all data 
				points
		Return:
			feature_probabilities: a dictionary whose entry (v, c) has the 
				computed probability of observing value 'v' among examples of
				class 'c', that is, P(X_j = c | Y = c).
				Note: v and c must be strings, the stored value must be float.
		"""

		feature_probabilities = {}

		## >>> YOUR CODE HERE >>>
		...
		## <<< END OF YOUR CODE <<<
		return feature_probabilities


	def fit(self, X_train, y_train):
		"""
		Fit Naive Bayes Classifier to the given data.

		This function computes all the necessary probability tables and stores
		them as dictionaries in the class.

		Input:
			X_train: a 2D numpy array, with string values, corresponding to the 
				pre-processed dataset
			y_train: a 1D numpy array, with string values, corresponding to the 
				pre-processed dataset

		Return:
			None
		"""
		n, d = X_train.shape
		self.d = d

		# store the class labels in a list, with a fixed order
		self.class_labels = np.array(list(set(y_train)))


		self.class_probs = self.compute_class_probability(y_train)
		self.feature_probs = []
		for j in range(d):
			Xj = X_train[:, j]
			self.feature_probs.append(self.compute_feature_probability(Xj, y_train))

		return

	def predict_probabilities(self, X_test):
		"""
		Input: X_test - 2D numpy array corresponding to the X for the test data
		Return: 
			probs - 2D numpy array with predicted probability for all classes, 
				for all test data points
		Objective: For the test data, compute posterior probabilities
		"""
		probs = np.zeros((len(X_test), len(self.class_labels)))
		## >>> YOUR CODE HERE >>>
		...
		## <<< END OF YOUR CODE <<<
		return probs

		
	def predict(self, probs):
		"""Get predicted label from a matrix of posterior probabilities
		
		Input:
			probs: 2D numpy array with predicted probabilities
		Return:
			y_pred: 1D numpy array with predicted class labels (strings), based
				on the probabilities provided
		"""
		y_pred = ...
		
		## >>> YOUR CODE HERE >>>
		...
		## <<< END OF YOUR CODE <<<
		
		return y_pred
		

		
	def evaluate(self, y_test, probs):
		"""
		Compute the 0-1 loss and squared loss for the predictions

		Input: 
			y_test: true labels of test data
			probs: predicted probabilities from `predict_proba`
		Return:
			0-1 loss and squared loss( See homework pdf for
				their mathematical definition)
		"""
		## >>> YOUR CODE HERE >>>
		zero_one_loss = ...
		squared_loss = ...
		## <<< END OF YOUR CODE <<<
		return (zero_one_loss, squared_loss)
	
def plot_nbc_curve(nbc, X, y):
	"""
	For each p in training_percentages[0.1,0.25, 0.35, 0.5, 0.75,0.9], 
	split the X_train, X_test, y_train, y_test using my_train_test_split function from utils.py
		and pass the parameter test_pct as 1 - p.
	Then using the X_train, X_test, y_train, y_test obtained, 
		calculate the training and test accuracy for each \verb|p|.
	Plot the training_accuracies and test_accuracies in y-axis and training_percentages*100 in x-axis.
	Save the plotted figure as "learning_curve_nbc.png" 
	
	Input:
		nbc: the Naive Bayes classifier
		X: data
		y: labels
		
	Return: None
	"""
	fig, ax = plt.subplots(1, figsize=(6, 4), constrained_layout=True)
	training_percentages = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

	np.random.seed(47)	
	perm = np.random.permutation(len(X))
	X, y = X[perm], y[perm]

	training_accuracies = []
	test_accuracies = []

	## >>> YOUR CODE HERE >>>
	...
	## <<< END OF YOUR CODE <<<

	# Plot the errorbars for training/test accuracy
	ax.errorbar(training_percentages * 100, y=training_accuracies,
				capsize=0.1, fmt="x-", label="Training")
	ax.errorbar(training_percentages * 100, y=test_accuracies,
				capsize=0.1, fmt="o--", label="Test")
	ax.legend()
	ax.set_xlabel("Training Set Size (%)")
	ax.set_ylabel("Accuracies")
	ax.set_title("Learning curves for NBC")
	fig.savefig(os.path.join(os.path.dirname(__file__), "learning_curve_nbc.png"))


"""
-------------------------------------------------------------------------------------------
THE CODE BELOW IS FOR EVALUATION. PLEASE DO NOT CHANGE!
-------------------------------------------------------------------------------------------
"""
import os

if __name__ == '__main__':

    os.system('cls' if os.name == 'nt' else 'clear')

    X, y = load_data(os.path.join(
        os.path.dirname(__file__), 'dataset/dating_train.csv'))
    X_train, X_valid, y_train, y_valid = my_train_test_split(
        X, y, 0.2, random_state=42)

    # Initialize and train a Naive Bayes classifier
    print('\n\n-------------Fitting NBC-------------\n')
    alpha = 1
    nbc = NaiveBayesClassifier(alpha)
    nbc.fit(X_train, y_train)
    print('\tDone fitting NBC.')

    print('\n\n-------------Naive Bayes Performace-------------\n')
    probs = nbc.predict_probabilities(X_train)
    nbc.evaluate(y_train, probs)
    p_train = nbc.predict(probs)
    print('Train Accuracy: ',accuracy( y_train, p_train))

    probs = nbc.predict_probabilities(X_valid)
    nbc.evaluate(y_valid, probs)
    p_valid = nbc.predict(probs)
    print('Validation Accuracy: ',accuracy( y_valid, p_valid))

    print('\n\n-------------Plotting learning curves-------------\n')

    print('Plotting Naive Bayes learning curves...')

    nbc.plot_nbc_curve(X, y)

    print('\n\nDone.')
	

   


  






    

    

    
    
    

    
