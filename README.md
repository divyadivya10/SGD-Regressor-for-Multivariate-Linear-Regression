# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load Dataset: Fetch the California Housing dataset using fetch_california_housing() and convert it into a pandas DataFrame.

2.Inspect Data: Print the first few rows of the DataFrame to understand the structure of the dataset and the target variables.

3.Separate Features and Target Variables: Split the dataset into feature variables (X) and target variables (Y), where Y includes both AveOccup and HousingPrice.

4.Split Data into Training and Test Sets: Use train_test_split() to split the data into training and test sets (e.g., 80% for training and 20% for testing).

5.Scale the Features and Target Variables: Apply StandardScaler to scale both the feature data (X) and the target variables (Y) to have zero mean and unit variance.

6.Initialize the Model: Create an instance of SGDRegressor for performing stochastic gradient descent, setting maximum iterations and tolerance values.

7.Wrap Model with MultiOutputRegressor: Use MultiOutputRegressor to enable the SGDRegressor to handle multiple target outputs (AveOccup and HousingPrice).

8.Train the Model: Fit the model on the scaled training data (X_train, Y_train).

9.Make Predictions: Predict the target values (AveOccup and HousingPrice) on the test set (X_test) using the trained model.

10.Evaluate the Model: Inverse transform the predictions and true target values back to their original scale, then calculate the Mean Squared Error (MSE) to evaluate the model’s performance.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by:Divya R
RegisterNumber: 212222040040
*/
```
```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
dataset=fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())

X = df.drop(columns=['AveOccup','HousingPrice'])
Y = df[['AveOccup','HousingPrice']]
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)
sgd = SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred = multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse= mean_squared_error(Y_test,Y_pred)
print("Mean Squared Error:",mse)
print("\nPredictions:\n",Y_pred[:5])
print("Name: Divya R")
print("Register No: 212222040040")
```

## Output:

## df.head()
![image](https://github.com/user-attachments/assets/6944112c-a261-460f-ae69-7c4c63fc7665)
## Prediction
![image](https://github.com/user-attachments/assets/a2e1fe7d-e9c9-4ff0-bb10-635fa515aae6)




## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
