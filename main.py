'''
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def load_data():
    train_dataset = h5py.File('trainset.hdf5', "r")
    print(train_dataset)
    x_train = np.array(train_dataset["X_train"][:])  # your train set features
    y_train = np.array(train_dataset["Y_train"][:])  # your train set labels

    test_dataset = h5py.File('testset.hdf5', "r")
    x_test = np.array(test_dataset["X_test"][:])  # your train set features
    y_test = np.array(test_dataset["Y_test"][:])  # your train set labels
    print('hi')
    return x_train, y_train, x_test, y_test


X_train, y_train, X_test, y_test = load_data()
print(X_train.shape)
print(y_train.shape)
print(np.unique(y_train, return_counts=True))
'''
# importing all the modules
import pandas as pd
import ast
import pickle
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn import linear_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
print('hi')
dfr=pd.read_csv('Train.csv')
dft=pd.read_csv('Test.csv')
dfs=pd.read_csv('sample_submission.csv')
print('*************train***************')
print(dfr.head())
print(dfr.describe())
print(dfr.info())
print(dfr.Gender.value_counts())

print('*************test***************')
print(dft.head())
print(dft.describe())
print(dft.info())
print('*************submission***************')
print(dfs.head())
print(dfs.describe())
print(dfs.info())
#types of our features
print("types of our features")
print('Measures are: ',type(dfr.Measures[0]))
print('Weights are: ',type(dfr.Weights[0]))
print('Genders are: ',type(dfr.Gender[0]))
print("types of our my label")
print('Ages are:',type(dfr.Age[0]))
# Apply the conversion function to the entire column
# Define a function to convert string tuples to actual tuples of floats
# data processing
print('data processing')
def convert_to_float_tuple(string_tuple):
    return ast.literal_eval(string_tuple)
dfr['float_tuple_column'] = dfr['Measures'].apply(convert_to_float_tuple)
print(dfr['float_tuple_column'])
print(type(dfr['float_tuple_column'][0][0]))
dfr[['element_1', 'element_2', 'element_3']] = pd.DataFrame(dfr['float_tuple_column'].tolist(), index=dfr.index)

print(dfr['element_1'])
print(type(dfr['element_1'][0]))
print(dfr.columns)

dfr['float_tuple_column1'] = dfr['Weights'].apply(convert_to_float_tuple)
print(dfr['float_tuple_column1'])
dfr[['welement_1', 'welement_2', 'welement_3']] = pd.DataFrame(dfr['float_tuple_column1'].tolist(), index=dfr.index)

print(dfr['welement_1'])
print(dfr.columns)

dft['float_tuple_column'] = dft['Measures'].apply(convert_to_float_tuple)
print(dft['float_tuple_column'])
dft[['element_1', 'element_2', 'element_3']] = pd.DataFrame(dft['float_tuple_column'].tolist(), index=dft.index)

print(dft['element_1'])
print(dft.columns)

dft['float_tuple_column1'] = dft['Weights'].apply(convert_to_float_tuple)
print(dft['float_tuple_column1'])
dft[['welement_1', 'welement_2', 'welement_3']] = pd.DataFrame(dft['float_tuple_column1'].tolist(), index=dft.index)

print(dft['welement_1'])
print(dft.columns)

le = LabelEncoder()
dfr['Gender'] = le.fit_transform(dfr['Gender'])
dft['Gender'] = le.fit_transform(dft['Gender'])

columns_to_drop = ['Measures', 'Weights','float_tuple_column1','float_tuple_column']
dfr = dfr.drop(columns=columns_to_drop)
print(dfr.columns)
print(dfr.head())
columns_to_drop = ['Measures', 'Weights','float_tuple_column1','float_tuple_column']
dft = dft.drop(columns=columns_to_drop)
print(dft.columns)
print(dft.head())

X_train = dfr.drop(['Age'], axis=1)


y_train = dfr['Age'].astype(int)
print(X_train)
print(y_train)
X_Train, X_Val, y_Train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=42)
print(X_Train)
print(y_Train)
# create model
lasso_model = Lasso(alpha=1.0)  # Alpha is the regularization strength
Des_model = DecisionTreeRegressor(random_state=42)
Rand_model = RandomForestRegressor(n_estimators=200,max_depth=9,random_state=42)  # Number of trees in the forest
svr_model = SVR(kernel='linear')  # You can choose different kernel functions
gar_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)  # Number of boosting stages and learning rate
lin_model=linear_model.LinearRegression() # y=mx+b

# fitting model
lasso_model.fit(X_Train, y_Train)
Des_model.fit(X_Train, y_Train)
Rand_model.fit(X_Train, y_Train)
svr_model.fit(X_Train, y_Train)
gar_model.fit(X_Train, y_Train)
lin_model.fit(X_Train, y_Train)

#predict
lasso_pred = lasso_model.predict(X_Val)
Des_pred=Des_model.predict(X_Val)
Rand_pred=Rand_model.predict(X_Val)
svr_pred=svr_model.predict(X_Val)
gar_pred=gar_model.predict(X_Val)
lin_pred=lin_model.predict((X_Val))
# mse
print('********msr********')
lasso_mse=mean_squared_error(y_val, lasso_pred)
print(lasso_mse)
Des_mse=mean_squared_error(y_val, Des_pred)
print(Des_mse)
Rand_mse=mean_squared_error(y_val,Rand_pred)
print(Rand_mse)
svr_msr=mean_squared_error(y_val,svr_pred)
print(svr_msr)
gar_msr=mean_squared_error(y_val,gar_pred)
print(gar_msr)
lin_msr=mean_squared_error(y_val,lin_pred)
print(lin_msr)
print('make it more efficient')
min=40
for i in tqdm(range(30)):
  X_Train, X_Val, y_Train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
  Rand_model = RandomForestRegressor(n_estimators=200, max_depth=9, random_state=42)
  Rand_model.fit(X_Train,y_Train)
  Rand_mse = mean_squared_error(y_val, Rand_pred)
  if Rand_mse<min:
      min= Rand_mse
      with open('studentmodelrand.pickle', 'wb') as f:
          pickle.dump(Rand_model, f)  # saves our model in our directorie in a pickel file
      print(min)
pickle_in=open('studentmodelrand.pickle','rb')
linear112=pickle.load(pickle_in)

submi_pred=linear112.predict(dft)
print(mean_squared_error(submi_pred,dfs.Age))

