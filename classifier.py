import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

def get_nn_model():
	nn_model = Sequential()

	#Input layer -> Hidden layer
	#Input size is 28 * 28 = 784
	#Sigmoid activation
	nn_model.add(Dense(128, \
		input_dim = 784, \
		activation = "sigmoid"))
	#Hidden layer -> Output layer
	#Softmax activation
	nn_model.add(Dense(10, \
		activation = "softmax"))

	#Compile the model
	#Use Mean Squared Error loss function
	nn_model.compile(loss = "MSE", \
		optimizer = "adam", \
		metrics = ["accuracy"])
	return nn_model

def get_cnn_model():
	cnn_model = Sequential()

	#Input layer
	#Input size is 28 * 28
	#Sigmoid activation
	cnn_model.add(Conv2D(32, \
		kernel_size = (4, 4), \
		activation = "sigmoid",
		input_shape = (28, 28, 1)))
	cnn_model.add(MaxPooling2D(pool_size = (2, 2)))

	cnn_model.add(Conv2D(64, \
		kernel_size = (4, 4), \
		activation = "sigmoid"))
	cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
	cnn_model.add(Flatten())

	#Intermediate hidden layer
	cnn_model.add(Dense(32, activation = "sigmoid"))
	#Last hidden layer
	#Softmax activation
	cnn_model.add(Dense(10, activation = "softmax"))

	#Compile the model
	#Use Mean Squared Error loss function
	cnn_model.compile(loss = "MSE", \
		optimizer = "adam", \
		metrics = ['accuracy'])
	return cnn_model

def convert_features_to_4darray(features_2d):
	ret_4d = []
	for sample in features_2d:
		sample_reshaped = np.reshape(sample, (28, 28))
		# print(sample_reshaped)
		sample_reshaped_3d = np.expand_dims(sample_reshaped, axis=2)
		#print(sample_reshaped_3d)
		ret_4d.append(sample_reshaped_3d)
	return np.array(ret_4d)

run_mode = 1

batch_size = 5
loss_function = MSE
num_epochs = 200
optimizer = Adam()
verbosity = 1
num_folds = 5

#Load data
train_dataframe = pd.read_csv("MNIST.csv")
train_dataset = train_dataframe.values

#Get features
train_features = train_dataset[:, 1:]
train_features_scaled = np.true_divide(train_features, 255)

train_features_4d = convert_features_to_4darray(train_features_scaled)

#Get labels
train_labels = train_dataset[:, 0]
#print(train_labels)
encoder = LabelEncoder()
encoder.fit(train_labels)
apply_encoder = encoder.transform(train_labels)
train_labels_encoded = np_utils.to_categorical(apply_encoder)
#print(train_labels_encoded)
#print(train_labels_encoded.shape)

accuracies_per_fold = []
losses_per_fold = []
histories_per_fold = []

#Define K-fold cross validator
kfold = KFold(n_splits = num_folds, shuffle = True)

#Begin k-fold cross validation model
if run_mode == 1:
	kfold_split = kfold.split(train_features_4d, train_labels_encoded)
else:
	kfold_split = kfold.split(train_features_scaled, train_labels_encoded)
	print(kfold_split)

for train, test in kfold_split:
	if run_mode == 1:
		#Compile and run the CNN Model
		model = get_cnn_model()

		model.compile(loss = loss_function, \
			optimizer = optimizer, \
			metrics = ['accuracy'])

		history = model.fit(train_features_4d[train], \
			train_labels_encoded[train], \
			epochs = num_epochs, \
			verbose = verbosity)
		scores = model.evaluate(train_features_4d[test], \
			train_labels_encoded[test], \
			verbose = verbosity)

		accuracies_per_fold.append(scores[1] * 100)
		losses_per_fold.append(scores[0])
		histories_per_fold.append(history)

	else:
		#Compile and run the NN model
		model = get_nn_model()

		model.compile(loss = loss_function, \
			optimizer = optimizer, \
			metrics = ['accuracy'])

		#Fit data to model
		history = model.fit(train_features_scaled[train], \
			train_labels_encoded[train], \
			epochs = num_epochs, \
			verbose = verbosity)
		scores = model.evaluate(train_features_scaled[test], \
			train_labels_encoded[test], \
			verbose = verbosity)

		accuracies_per_fold.append(scores[1] * 100)
		losses_per_fold.append(scores[0])
		histories_per_fold.append(history)

fold_number = 1
for fold_number in range(num_folds):
	this_history = histories_per_fold[fold_number - 1]
	if run_mode == 1:
		plot_title = f"CNN Model: Fold {fold_number} Loss"
	else:
		plot_title = f"NN Model: Fold {fold_number} Loss"
	plt.plot(range(len(this_history.history['loss'])), this_history.history['loss'] )
	plt.title(plot_title)
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.show()

	if run_mode == 1:
		plot_title = f"CNN Model: Fold {fold_number} Accuracy"
	else:
		plot_title = f"NN Model: Fold {fold_number} Accuracy"
	plt.plot(range(len(this_history.history['accuracy'])), this_history.history['accuracy'] )
	plt.title(plot_title)
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.show()

fold_number = 1
for acc in accuracies_per_fold:
	print(f"Fold {fold_number} Accuracy: {acc}")
	fold_number += 1
mean_accuracy = sum(accuracies_per_fold) / num_folds
print(f"Mean accuracy: {mean_accuracy}")

print("Done!")
