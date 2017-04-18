# Make a prediction with weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			sum_error += error**2
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
			
		
		if epoch == n_epoch-1:
			print('Training Info - epoch=%d, lrate=%.3f, error=%.3f\n' % (epoch+1, l_rate, sum_error))
			if sum_error > 0:
				 print("Cannot Learn with this perceptron\n")
			else:
				print("Successfully Learned\n")
	return weights



	


and3 = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]]
or3 = [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 1], [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]]
and2 = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]
or2 = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
nand2 = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
nand3 = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 1], [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
nor2 = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0]]
nor2 = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 0]]
not1 = [[0,1],[1,0]]
xor3 = [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]]
xor2 = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
xnor2 = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]]
xnor3 = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]


datasets = [and3 ,or3 ,and2 ,or2 ,nand2 ,nand3 ,nor2 ,nor2 ,not1 ,xor3 ,xor2 ,xnor2 ,xnor3 ]
dataname = ["and3","or3" ,"and2" ,"or2" ,"nand2" ,"nand3" ,"nor2" ,"nor2" ,"not1" ,"xor3" ,"xor2" ,"xnor2" ,"xnor3" ]
def x(dataset):
	#dataset = [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]]
	l_rate = 0.1
	n_epoch = 20
	weights = train_weights(dataset, l_rate, n_epoch)
	print(weights)

	for row in dataset:
		prediction = predict(row, weights)
		print("Expected=%d, Predicted=%d" % (row[-1], prediction))
		
	print("\n\n\n")
i=0
for dataset in datasets:
	print("\n\nDATA -> "+dataname[i]+"")
	i = i+1
	x(dataset)
	


	


