import pickle

# load the model from disk
loaded_model = pickle.load(open('pickle_iris_flowers_finalized_model.sav', 'rb'))

# define one new instance
Xnew = [[5.4, 3.4, 1.7, 0.2], [5.5, 2.3, 4.0, 1.3], [5.6, 2.8, 4.9, 2.0]]

# make a prediction
ynew = loaded_model.predict(Xnew)

# show the inputs and predicted outputs
for i in range(len(Xnew)):
    print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
