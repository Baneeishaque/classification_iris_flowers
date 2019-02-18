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

# define one new instance
Xnew = [[5.4, 3.4, 1.7, 0.2], [5.5, 2.3, 4.0, 1.3], [5.6, 2.8, 4.9, 2.0]]

# make a prediction
ynew = knn.predict(Xnew)

# show the inputs and predicted outputs
for i in range(len(Xnew)):
    print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
