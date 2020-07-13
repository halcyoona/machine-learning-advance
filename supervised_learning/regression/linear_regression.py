import numpy as np
import matplotlib.pyplot as plt


X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)


plt.figure(figsize=(16, 8))
plt.scatter(
    X,
    y,
    c='black'
)
plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales ($)")
plt.show()


X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance

# calculating theta
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# making prediction
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)



plt.figure(figsize=(16, 8))
plt.scatter(
    X,
    y,
    c='black'
)
plt.plot(
    X_new,
    y_predict,
    c='blue',
    linewidth=2
)

plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales ($)")
plt.show()



#calculating MSE i.e cost function or error function
y_new = np.array([[0], [2]])
y_new[0][0] = 4 + 3 * X_new[0][0]
y_new[1][0] = 4 + 3 * X_new[1][0]

MSE = 0
for i in range(2):
    MSE += ((y_predict[i][0] - y_new[i][0])**2)
MSE = MSE / 2