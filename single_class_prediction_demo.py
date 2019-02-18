# example of making a single class prediction
from sklearn.datasets.samples_generator import make_blobs
from sklearn.linear_model import LogisticRegression


def generate_data_frame():
    return make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)


def get_final_lr_model(current_x, current_y):
    current_model = LogisticRegression()
    current_model.fit(current_x, current_y)
    return current_model


# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)

# fit final model
model = LogisticRegression()
model.fit(X, y)

# define one new instance
Xnew = [[-0.79415228, 2.10495117]]

# make a prediction
ynew = model.predict(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
