# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the linear regression using gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: M.Suwetha
RegisterNumber:  212221230112
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("ex1.txt",header=None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
    """"
    Take in a numpy array X,y,theta and generate the cost function of using theta as a parameter in a linera regression tool   
    """
    m=len(y) 
    h=X.dot(theta) 
    square_err=(h-y)**2
    return 1/(2*m)*np.sum(square_err) 

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) 

def gradientDescent(X,y,theta,alpha,num_iters):
    """"
    Take in numpy array X,y and theta and update theta by taking num_iters gradient steps with learning rate of alpha 
    return theta and the list of the cost of the theta during each iteration
    """
    m=len(y)
    J_history=[] #empty list
    for i in range(num_iters):
        predictions=X.dot(theta)
        error=np.dot(X.transpose(),(predictions-y))
        descent=alpha*(1/m)*error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    """"
    Takes in numpy array of x and theta and return the predicted valude of y based on theta
    """
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For Population = 35000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For Population = 70000, we predict a profit of $"+str(round(predict2,0)))
*/
```

## Output:
![193342362-8c2bc825-5f6c-49cc-b58f-a2cacc4b3782](https://user-images.githubusercontent.com/94165336/197791590-74e9e0d8-c3f1-4abf-bb95-58c703577b94.png)
![193342367-20be0b59-2813-4d7b-a5e1-83b8dbf8e6d1](https://user-images.githubusercontent.com/94165336/197791744-ea0cd132-5f61-47f6-9b11-00f4ceae97a4.png)
![193342369-faff![193342370-e2e9d9fd-1b2e-4a92-8821-6b4714fad6bc](https://user-images.githubusercontent.com/94165336/197791893-d985ec89-ff8e-4def-b022-b3af3c9d7ce5.png)
7d40-aea9-45a4-90d8-cd677adb57c5](https://user-images.githubusercontent.com/94165336/197791822-e00ee990-b8ed-41c1-86a1-4faa2b822ada.png)
![193342373-d6296f86-5c1b-4d23-819b-a367784628a0](https://user-images.githubusercontent.com/94165336/197791945-89c0641d-4071-4887-84b7-24bf2ca2e7e5.png)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
