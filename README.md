# Step 1 : import library
import pandas as pd


# Step 2 : import data
default = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Credit%20Default.csv')


default.head()
default.info()
default.describe()
default['Default'].value_counts()
# Step 3 : define target (y) and features (X)


default.columns
y = default['Default']
X = default.drop(['Default'],axis=1)


# Step 4 : train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)


# check shape of train and test sample
X_train.shape, X_test.shape, y_train.shape, y_test.shape

((1400, 4), (600, 4), (1400,), (600,))

# Step 5 : select model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# Step 6 : train or fit model
model.fit(X_train,y_train)
