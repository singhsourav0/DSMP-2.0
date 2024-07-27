import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from scratch import meraknn

df = pd.read_csv('C:/Users/Dell/python/PROJECT/machine_learning/KNN/Social_Network_Ads.csv')
df = df.iloc[:, 1:]

encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])
scaler = StandardScaler()

x = df.iloc[:, 0:3].values
x = scaler.fit_transform(x)
y = df.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x, y)

y_pred = knn.predict(x_test)
print(accuracy_score(y_test, y_pred))

#print(y)
#type(df)
apnaknn = meraknn(k=5)

apnaknn.fit(x_train,y_train)
apnaknn.predict(x_test)

