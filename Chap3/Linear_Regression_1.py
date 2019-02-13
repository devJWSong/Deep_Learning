import numpy as np

X = [2, 4, 6, 8]
y = [81, 93, 91, 97]

mX = np.mean(X)
my = np.mean(y)

print("X mean: ", mX)
print("y mean: ", my)

divisor = sum([(mX-i)**2 for i in X])

def top(X, mX, y, my):
    d = 0
    for i in range(len(X)):
        d += (X[i] - mX) * (y[i] - my)
    return d

dividend = top(X, mX, y, my)

a = dividend /divisor
b = my - (mX * a)

print("a = ", a)
print("b = ", b)