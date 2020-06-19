# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:50:47 2020

@author: User
"""


# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# Import Dataset
datasetskripsi = pd.read_csv('data2.csv')

# Transform the dataset into a data frame
mydata = pd.DataFrame(datasetskripsi)
df_x = pd.DataFrame(mydata.Penggunaan)
df_y = pd.DataFrame(mydata.Pemesanan)

# Eksplorasi Data
  # visualisasi data
plt.scatter(df_x, df_y)
plt.xlabel("Penggunaan")
plt.ylabel("Pemesanan")
plt.title("Penggunaan vs Pemesanan Reagent Kimia")
plt.show()

sns.regplot(df_x, df_y);

#Mengetahui nilai korelasi dari penggunaan dan pemesanan.
mydata.corr()
#Nilai korelasinya adalah 0.88 termasuk kategori sangat tinggi.

  # Get some statistics from the data set, count, mean
df_x.describe()
df_y.describe()

# Initialize the linear regression model
reg = linear_model.LinearRegression()

# Split the data into 80% training and 20% testing data (prinsip pareto)
x_train, x_test, y_train, y_test = train_test_split(
    df_x, df_y, test_size = 0.2)

# Train the model with our training data
reg.fit (x_train, y_train)

# Print the coefecients (nilai m)
print(reg.coef_)

# Print the intercepts (nilai b)
print(reg.intercept_)

# y = mx + b
# Dari nilai m dan b diatas, jika terapkan ke dalam rumus menjadi:
print("y = ", (reg.coef_),"x + ", (reg.intercept_))

# Print the predictions on our test data
y_pred = reg.predict(x_test)
print (y_pred) # ini menampilkan data prediksi oleh model machine learning bisa di banding kan dengan y_test

# Print the actual values
print(y_test) # ini menampilkan data testing yang 20%

# Mencari akurasi score dari model yang terbentuk menggunakan data testing yang telah di split
reg.score(x_test, y_test)

from numpy import mean, absolute

# Check the model performance / accuracy using Mean Absolute Deviation (MAD)
print ("Mean Absolute Deviation (MAD) = ", mean(
    absolute(y_pred - y_test - mean(y_pred - y_test))))

# Check the model performance / accuracy using Mean Squared Error (MSE)
print ("Mean Square Error (MSE) = ", (mean(y_pred - y_test)**2))

# Check the model performance / accuracy using Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_test, y_pred): 
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print ("Mean Absolute Percentage Error (MAPE) = ", (
    mean_absolute_percentage_error(y_test, y_pred)))

#Visualizing the Training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, reg.predict(x_train), color = 'green')
plt.title('Data Training Penggunaan dan Pemesanan Reagent Kimia')
plt.xlabel('Penggunaan')
plt.ylabel('Pemesanan')
plt.show()


#Visualizing the Test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, reg.predict(x_train), color = 'green')
plt.title('Data Testing Penggunaan dan Pemesanan Reagent Kimia')
plt.xlabel('Penggunaan')
plt.ylabel('Pemesanan')
plt.show()

# Deployment
# Prediksi jumlah untuk memesan reagent (apa namanya sebutin disini) dengan penggunaan sebelumnya 11
reg.predict([[11]])