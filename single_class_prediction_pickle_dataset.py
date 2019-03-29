import pickle

from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier

import dataset_preparation

dataset = dataset_preparation.get_data_frame()

# Split-out validation dataset
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.20
seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)

# save the model to disk
filename = 'pickle_iris_flowers_finalized_model.sav'
pickle.dump(knn, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

# define one new instance
Xnew = [[5.4, 3.4, 1.7, 0.2]]

# make a prediction
ynew = loaded_model.predict(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
