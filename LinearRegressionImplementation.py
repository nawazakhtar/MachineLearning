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

