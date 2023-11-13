import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
df = pd.read_csv('Flight_Price_Dataset_Q2.csv')
dummies = pd.get_dummies(df[['departure_time', 'stops', 'arrival_time', 'class']])
x = df[['duration', 'days_left']]
y = df['price']
x = pd.concat([x, dummies], axis=1)

x['arz_mabda'] = 1
column = x.columns
x = np.nan_to_num(x)
y = np.nan_to_num(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

theta = np.zeros(X_train.shape[1])
alpha = 0.002
num_iters = 12000
start = time.time()
for i in range(num_iters):
    h = np.dot(X_train, theta)
    error = h - y_train
    grad = np.dot(X_train.T, error) / len(y_train)
    theta -= alpha * grad
end = time.time()

y_pred = np.dot(theta, X_test.T)
s = "Price = "
for i in range(len(column) - 1):
  s += f"({theta[i]}) * {column[i]} + "
s += str(theta[-1])
s += f"""
Training Time: {end - start}s

Logs:
MSE: {mean_squared_error(y_test, y_pred)}
RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}
MAE: {mean_absolute_error(y_test, y_pred)}
R2: {r2_score(y_test, y_pred)}"""
with open('14-UIAI4021-PR1-Q2.txt', 'w') as file:
  file.write(s)