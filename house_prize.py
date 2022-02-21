from houseprice.models import housedata
import joblib
import pandas as pd
import numpy as np
import csv

Data = pd.read_csv('kc_house_data.csv')

Data = Data.drop('date',axis=1)
Data = Data.drop('id',axis=1)
Data = Data.drop('zipcode',axis=1)
with open('kc_house_data.csv') as csvfile:
     reader = csv.DictReader(csvfile)
     for row in reader:
        p = housedata(Bedroom=row['bedrooms'], Bathroom=row['bathrooms'],sqft_living=row['sqft_living'],sqft_lot=row['sqft_lot'],floors=row['floors'],wanterfront=row['row'],view=row['view'],condation=row['condation'],grade=row['grade'])
housedata.objects.create(p)

X = Data.drop('price',axis =1).values
y = Data['price'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

from sklearn.preprocessing import StandardScaler
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(float))
X_test = s_scaler.transform(X_test.astype(float))

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20, random_state=0)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))  
print('MSE:', metrics.mean_squared_error(y_test, y_pred))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('VarScore:',metrics.explained_variance_score(y_test,y_pred))
print(metrics.r2_score(y_test,y_pred))

fileName="finalModel.sav"
joblib.dump(regressor,fileName)