###k armed bandit

import numpy as np
import ipdb





k=10

totalReward = 0 
means  = np.random.rand(k,1)
arms  = np.zeros(shape=(k, 1000))
for i in range(k):
	arms [i] = np.random.normal(means[i], 1, 1000)

epsilons = [0,0.01,0.1]
for eps in epsilons:
	qvalues = np.zeros((k,1)) 
	numberOfVisits = np.zeros((k,1))


	for i in range(1000):
		currentChoice = np.random.choice([0,1],1,p = [1-eps, eps])
		if currentChoice==1:
			arm  = np.random.choice(k)

		elif currentChoice==0:
			arm  = np.argmax(qvalues)

		#draw a sample from arm and then update its qvalue 
		reward = np.random.choice(arms[arm])
		numberOfVisits[arm] +=1

		qvalues[arm] += (1/numberOfVisits[arm]) * (reward - qvalues[arm])


