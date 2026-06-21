import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn import svm 
from sklearn.metrics import accuracy_score 
import pickle 

# Load details 
df = pd.read_csv("diabetes.csv")
X = df.drop(columns= 'Outcome', axis= 1)
Y = df['Outcome']

# scale the data 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train test split 
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size =0.2, stratify = Y, random_state=2)

# train and evaluate SVM 
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train, Y_train)
svm_predictions = svm_model.predict(X_test)
svm_acc = accuracy_score(Y_test, svm_predictions)

# train and evaluate LR 
lr_model = LogisticRegression()
lr_model.fit(X_train, Y_train)
lr_predictions = lr_model.predict(X_test)
lr_acc = accuracy_score(Y_test, lr_predictions)

# compare outputs 
print("-" * 30 )
print(f"SVM Accuracy:                        {svm_acc * 100:.2f}%")
print(f"Logistic Regression Accuracy:        {lr_acc * 100:.2f}%")
print("-" * 30 )

# save the LR model and scaler 

with open('lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Logistic Regression Model and Scaler saved for deployment")

