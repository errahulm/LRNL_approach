import model1
import model2
import model3
from sklearn.metrics import accuracy_score
from numpy import array
import pandas as pd
import numpy as np
from numpy import argmax
from numpy import tensordot
from numpy.linalg import norm
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings("ignore")

testy=pd.read_csv("y_test.txt", sep=",", header=None)

for i in range(len(testy)):
    testy.iloc[i]=testy.iloc[i]-1
testy=np.array(testy)   
 
# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, weights):
    yhats = [ model1.modelp1(), model2.modelp2(), model3.modelp3()]
    yhats = array(yhats)
	# weighted sum across ensemble members
    summed = tensordot(yhats, weights, axes=((0),(0)))
	# argmax across classes
    result = argmax(summed, axis=1)
    return result

# # evaluate a specific number of members in an ensemble
def evaluate_ensemble(members, weights):
	# make prediction
    yhat = ensemble_predictions(members, weights)
	# calculate accuracy
    return accuracy_score(testy, yhat)

# normalize a vector to have unit norm
def normalize(weights):
	# calculate l1 vector norm
	result = norm(weights, 1)
	# check for a vector of all zeros
	if result == 0.0:
		return weights
	# return normalized vector (unit norm)
	return weights / result

# loss function for optimization process, designed to be minimized
def loss_function(weights, members):
	# normalize weights
	normalized = normalize(weights)
	# calculate error rate
	return 1.0 - evaluate_ensemble(members, normalized)

members=[model1.modelf1(),model2.modelf2(),model3.modelf3()]

_, test_acc = model1.modele1()
print('Model %d: %.3f' % (1, test_acc))

_, test_acc = model2.modele2()
print('Model %d: %.3f' % (2, test_acc))

_, test_acc = model3.modele3()
print('Model %d: %.3f' % (3, test_acc))
# evaluate averaging ensemble (equal weights)
n_members = len(members)
weights = [1.0/n_members for _ in range(n_members)]
score = evaluate_ensemble(members, weights)
print('Equal Weights Score: %.3f' % score)

# define bounds on each weight
bound_w = [(0.0, 1.0)  for _ in range(n_members)]
print(bound_w)
# arguments to the loss function
search_arg = (members, testy)
#print(search_arg)

# global optimization of ensemble weights
result = differential_evolution(loss_function, bound_w, search_arg, maxiter=1000, tol=1e-7)
# get the chosen weights
weights = normalize(result['x'])

print('Optimized Weights: %s' % weights)
# evaluate chosen weights
score = evaluate_ensemble(members, weights, testy)
print('Optimized Weights Score: %.3f' % score)

