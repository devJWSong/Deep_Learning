import numpy as np

ab = [3, 76]

data = [[2, 81], [4, 93], [6, 91], [8, 97]]
X = [i[0] for i in data]
y = [i[1] for i in data]

def predict(X):
    return ab[0]*X + ab[1]

def rmse(p, a):
    return np.sqrt(((p-a)**2).mean())

def rmse_val(predic_result, y):
    return rmse(np.array(predic_result), np.array(y))

predict_result = []

for i in range(len(X)):
    predict_result.append(predict(X[i]))
    print("Study time=%f, Actual score=%f, Prediction score=%f" % (X[i], y[i], predict(X[i])))

print("rmse value: " + str(rmse_val(predict_result, y)))