from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report, f1_score, RocCurveDisplay
from sklearn.metrics import make_scorer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import StackingClassifier, VotingClassifier, AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load your dataset
df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

# Split the dataset into features (X) and target variable (y)
X = df.drop('Diabetes_012', axis=1)
y = df['Diabetes_012']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Apply SMOTE to balance the training data")
# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# creating a dataframe for storing model outputs
df_models_output = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'roc', 'f1_score'])

# defining a custom scorer for f1_score
f1_scorer = make_scorer(f1_score, average='macro')

# GRADIENT BOOSTING (LIGHTGBM) MODEL
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier()

# Random Undersampled data
#lgb_model.fit(X_undersample, y_undersample)

# SMOTE Oversampled data
#lgb_model.fit(X_train_SMOTE, y_train_SMOTE)

# SMOTE-ENN Oversampled data
lgb_model.fit(X_train_smote, y_train_smote)

# Prediction
y_pred = lgb_model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model to disk
with open('mnb_spam_detector.pkl', 'wb') as file:
    pickle.dump(lgb_model, file)

exit()



def compute_evaluation_metric(algo,X_test,y_actual,y_pred,y_pred_prob):
    cm=confusion_matrix(y_actual,y_pred,labels=y.value_counts().index)
    print(f"confusion matrix -\n {cm}\n")
    accuracy=accuracy_score(y_actual,y_pred)
    print(f"accuracy score : {accuracy}")
    
    
    TP=(np.diag(cm))
    
    FP=(cm.sum(axis=0)-np.diag(cm))
    #print(FP)
    FN=(cm.sum(axis=1)-np.diag(cm))
    #print(FN)
    #print(f"cm sum -{cm.sum()}")
    TN=cm.sum()-(FP+FN+TP)
    #print(TN)
    
    TPR=np.round(np.mean(TP/(TP+FN)),4)
    TNR=np.round(np.mean(TN/(TN+FP)),4)
    FPR=np.round(np.mean(FP/(FP+TN)),4)
    precision=(TP/(TP+FP))
    f1score = f1_score(y_test, y_pred, average='macro')
    print(f"Precision for positive class is {precision}")
    print(f"TPR/Recall is {TPR}")
    print(f"TNR/Specifity is {TNR}")

    print(f"FPR is {FPR}" )
    print(f"F1 score is - {f1score}")
    print(f"\n classification report - :\n {classification_report(y_actual,y_pred)}")
    ROC = roc_auc_score(y_actual, y_pred_prob, average='macro', multi_class='ovr')
    print(f"ROC -{ROC} ") # ROC curve method for binary classification problems
    return algo,accuracy,precision,TPR,ROC,f1score



# creating instances for all the classifiers
mnb = MultinomialNB(alpha=0.1)  # Multinomial Naive Bayes
svc = SVC(C=10, gamma=0.1, kernel='sigmoid', probability=True)  # Support Vector Machine
knn = KNeighborsClassifier()  # K-nearest Neighbors
rf = RandomForestClassifier(criterion='gini', max_features='sqrt', random_state=33)  # Random Forest
xgb = XGBClassifier(learning_rate=0.1, n_estimators=150, random_state=33)  # Extreme Gradient Boosting
adbst = AdaBoostClassifier(learning_rate=0.1, n_estimators=200, random_state=33)  # Adaptive Boosting

estimators=[('nb',mnb),('svc',svc),('xg',xgb)]
Vclf=VotingClassifier(estimators=estimators,voting='soft') #voting classifier

final_estimator=rf
sclf=StackingClassifier(estimators=estimators,final_estimator=final_estimator) #stcking classifier


#fit the train data of each model and save the output to dataframe
for modl in (mnb,svc,knn,rf,xgb,adbst,Vclf,sclf):
    print(type(modl).__name__)
    modl.fit(X_train_smote,y_train_smote)
    y_pred=modl.predict(X_test)
    y_pred_prob=modl.predict_proba(X_test)
    df_models_output.loc[len(df_models_output)]=compute_evaluation_metric(type(modl).__name__,X_test,y_test,y_pred,y_pred_prob)


df_models_output.sort_values(by=['accuracy','f1_score'],ascending=False)


# Save the model to disk
with open('mnb_spam_detector.pkl', 'wb') as file:
    pickle.dump(lgb, file)
    
    