
# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Store data in a structured format (e.g., CSV, DataFrame).
2. Use a Simple Linear Regression model to fit the training data.
3. Use the trained model to predict values for the test set.
4. Evaluate performance using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## Program:
~~~
Developed by: Varun A
RegisterNumber: 212224240178
~~~
~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
#displaying the content in datafile
df.head()
~~~
## Output:

![mlpic](https://github.com/user-attachments/assets/46372791-1b03-4fef-9bf9-21ac7f56215c)

~~~
df.tail()
~~~
## Output:
![image](https://github.com/user-attachments/assets/325b3fcd-343e-48de-be71-882eb4ffa902)

~~~
x=df.iloc[:,:-1].values
x
~~~
## Output:
![image](https://github.com/user-attachments/assets/f1e7c477-8729-4146-b542-028d779322e1)

~~~
y=df.iloc[:,1].values
y
~~~
## Output:
![image](https://github.com/user-attachments/assets/b24dd9e7-57c4-48b3-a8b0-e07b12b57973)


~~~
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
~~~
~~~
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
~~~

~~~
y_pred
~~~
## Output:
![image](https://github.com/user-attachments/assets/4e4bcaf3-7475-4961-9698-bb123e9a39fb)

~~~
y_test
~~~
## Output:
![image](https://github.com/user-attachments/assets/17e79a54-20e5-46b7-a147-3188fc659f53)

~~~
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE =',mae)
rmse=np.sqrt(mse)
print("RMSE =",rmse)
~~~
## Output:
![image](https://github.com/user-attachments/assets/e6125223-71e0-477a-8ab8-60524f53691e)
~~~
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
~~~
## Output:
![image](https://github.com/user-attachments/assets/0d169e89-8175-4a89-9c64-72766c520da9)

~~~
plt.scatter(x_test,y_test,color="orange")
plt.plot(x_test,y_pred,color="red")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
~~~
## Output:
![image](https://github.com/user-attachments/assets/5d417873-550b-4717-9d7c-a15da78c6862)





















## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
