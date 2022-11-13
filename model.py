import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

dataset=pd.read_csv('diabetes.csv')
X=dataset.drop(columns='Outcome',axis=1)
y=dataset['Outcome']

scaler=StandardScaler()
X=scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=2)

clf=svm.SVC(kernel="linear")
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
# accuracy=accuracy_score(y_pred,y_test)

# saving model to disk
pickle.dump(clf, open('model.pkl','wb'))
# load model
model = pickle.load(open('model.pkl','rb'))