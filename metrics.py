from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder #str to num
import pandas as pd

f=pd.read_csv("cutoff_clg.csv")    # min 5000 rows, max 10,000 rows
p1=LabelEncoder()   # top_colleges.csv==college_name
f["cast"]=p1.fit_transform(f["cast"])   # cutoff.csv==college_code,cast,cutoff,year
x=f.drop(["college_code","year"],axis=1)
y=f["college_code"]
x_train,xtest,y_train,ytest=train_test_split(x,y,test_size=0.3,random_state=0)
xtrain,xval,ytrain,yval=train_test_split(x_train,y_train,test_size=0.1111,random_state=0)
model=RandomForestClassifier(n_estimators=100,random_state=42,oob_score=True)
model.fit(xtrain,ytrain)
ypre=model.predict(xtest)

print("validation score:",accuracy_score(yval,model.predict(xval))*100)  # 60% to 90%
print("test score:",model.score(xtest,ytest)*100)   # 60% to 85%
print("train score:",model.score(xtrain,ytrain)*100)    #80% to 95%
print("Classification Report:\n",classification_report(ytest,ypre,zero_division=1)) 
print("Cross-Validation Scores:",cross_val_score(model,x,y,cv=5))
print("Average Accuracy:",(cross_val_score(model,x,y,cv=5)).mean())
print("OOB Score:",model.oob_score_)

ovr_model=OneVsRestClassifier(model)
ovr_model.fit(xtrain, ytrain)
score=roc_auc_score(pd.get_dummies(yval),ovr_model.predict_proba(xval),multi_class='ovr')
print("Multiclass AUC Score:",score)

misclassified=xtest.copy()
misclassified["actual_college_code"]=ytest.values
misclassified["predicted_college_code"]=ypre
misclassified_samples=misclassified[misclassified["actual_college_code"]!=misclassified["predicted_college_code"]]
print("Misclassified Samples:")
print(misclassified_samples.head(10)) 
print("Top Mispredicted Targets:")
print(misclassified_samples["predicted_college_code"].value_counts().head())
correct_samples=misclassified[misclassified["actual_college_code"]==misclassified["predicted_college_code"]]
print("Correct Predictions:",len(correct_samples))

cm=confusion_matrix(ytest,ypre)
sns.heatmap(cm,annot=False)
plt.title("Confusion Matrix(Test Set)")
plt.show()

cm=confusion_matrix(ytest,ypre,normalize='true')  
plt.figure(figsize=(14,10))  
sns.heatmap(cm,annot=False,cmap='viridis',cbar=True)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.show()

probs=model.predict_proba(xtest)
true_probs=probs.max(axis=1)  
true_classes=(ytest==ypre)
fraction_of_positives,mean_predicted_value=calibration_curve(true_classes,true_probs,n_bins=10)
plt.plot(mean_predicted_value,fraction_of_positives,marker='o')
plt.plot([0,1],[0,1],linestyle='--')
plt.title("Calibration Curve")
plt.xlabel("Mean Predicted Value")
plt.ylabel("Fraction of Positives")
plt.show()

feature_names = x.columns
importances = model.feature_importances_
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("RandomForest Feature Importance")
plt.show()
