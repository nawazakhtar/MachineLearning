#Implementation of Linear Regression Model using Gradient Descent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load the data
data = pd.read_csv('data.txt', names = ['Population','Profit'])
#print data
print data
#print data head means first five rows
print data.head()
#plot the data
X_df = pd.DataFrame(data.Population)
y_df = pd.DataFrame(data.Profit)
#number of observations
m = len(y_df)
#plotting the data in figure with x- label and y-label
plt.figure(figsize=(10,8))
plt.plot(X_df, y_df, 'kx')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()
#plotting the data with given data and (x,y) points
plt.figure(figsize=(10,8))
plt.plot(X_df, y_df, 'k.')
plt.plot([5, 22], [6,6], '-')
plt.plot([5, 22], [0,20], '-')
plt.plot([5, 15], [-5,25], '-')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()
#Implementing cost function
x_quad = [n/10 for n in range(0, 100)]
y_quad = [(n-4)**2+5 for n in x_quad]
#plotting the graph between x and f(x)
plt.figure(figsize = (10,7))
plt.plot(x_quad, y_quad, 'k--')
plt.axis([0,10,0,30])
plt.plot([1, 2, 3], [14, 9, 6], 'ro')
plt.plot([5, 7, 8],[6, 14, 21], 'bo')
plt.plot(4, 5, 'ko')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Quadratic Equation')
plt.show()
#Adding number of iterations to attain global minima and learning rate as alpha
iterations = 1500
alpha = 0.01
# Add a columns of 1s as intercept to X
X_df['intercept'] = 1
# Transform to Numpy arrays for easier matrix math and start theta at 0
X = np.array(X_df)
y = np.array(y_df).flatten()
theta = np.array([0, 0])
def cost_function(X, y, theta):
    """
    cost_function(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
    """
    ## number of training examples
    m = len(y) 
    
    ## Calculate the cost with the given parameters
    J = np.sum((X.dot(theta)-y)**2)/2/m
    
    return J
result = cost_function(X, y, theta)
print result
print('X is-->>>')
print X
print('Y is---->>>>')
print y
print('theta is ---->>')
print theta

def gradient_descent(X, y, theta, alpha, iterations):
    """
    gradient_descent Performs gradient descent to learn theta
    theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
    taking num_iters gradient steps with learning rate alpha
    """
    cost_history = [0] * iterations
    
    for iteration in range(iterations):
        hypothesis = X.dot(theta)
        print hypothesis
        loss = hypothesis-y
        print loss
        gradient = X.T.dot(loss)/m
        theta = theta - alpha*gradient
        cost = cost_function(X, y, theta)
        cost_history[iteration] = cost

    return theta, cost_history
(t, c) = gradient_descent(X,y,theta,alpha, iterations)
print t
print("gradient descent")

# Prediction
print np.array([3.5, 1]).dot(t)
print np.array([7, 1]).dot(t)

## Plotting the best fit line
best_fit_x = np.linspace(0, 25, 20)
best_fit_y = [t[1] + t[0]*xx for xx in best_fit_x]


plt.figure(figsize=(10,6))
plt.plot(X_df.Population, y_df, '.')
plt.plot(best_fit_x, best_fit_y, '-')
plt.axis([0,25,-5,25])
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Profit vs. Population with Linear Regression Line')
plt.show()




