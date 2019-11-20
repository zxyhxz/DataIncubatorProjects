# ---------------------------- SVM --------------------------------------
# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn import preprocessing
from pandas.core.frame import DataFrame
warnings.filterwarnings("ignore")


# %%-----------------------------------------------------------------------
# importing Data
# read data as pandas dataframe
data = pd.read_csv("cc_default_data_SVM.csv")

# define column names
data.columns = ['ID', 'LimitBalance', 'Sex', 'Education', 'MaritalStatus', 'Age', 'Repayment_Sept',
                'Repayment_Aug', 'Repayment_July', 'Repayment_June', 'Repayment_May', 'Repayment_Apr',
                'BillAmt_Sept', 'BillAmt_Aug', 'BillAmt_July', 'BillAmt_June','BillAmtMay', 'BillAmt_Apr',
                'PaymentAmt_Sept', 'PaymentAmt_Aug', 'PaymentAmt_July', 'PaymentAmt_June', 'PaymentAmt_May',
                'PaymentAmt_Apr', 'default payment next month']

# %%-----------------------------------------------------------------------
# Data pre-processing

# drop unnecessary rows and columns
data.drop([0], inplace=True)
data.drop(['ID'], axis=1, inplace=True)
# look at the first few rows
print(data.head())
# print(data.columns.tolist())


# replace missing characters as NaN
data = data.replace('?', np.NaN, inplace=False)
# check the structure of data
data.info()
# check the null values in each column
print(data.isnull().sum())
# check the summary of the data
data.describe(include='all')


# normalize continuous columns such as LimitBalance, BillAmount and PaymentAmount
X_Part1 = data.values[:, :1]
X_Part2 = data.values[:, 11:23]
X_Part5 = data.values[:, 4:5]
min_max_data = preprocessing.MinMaxScaler()

# transfer them into dataframe to merge with other parts
X_Part1_minmax = DataFrame(min_max_data.fit_transform(X_Part1))
X_Part2_minmax = DataFrame(min_max_data.fit_transform(X_Part2))
X_Part5_minmax = DataFrame(min_max_data.fit_transform(X_Part5))

# adjust index to be consistent with get_dummies dataframe below
X_Part1_minmax.index = range(1, len(X_Part1_minmax)+1)
X_Part2_minmax.index = range(1, len(X_Part2_minmax)+1)
X_Part5_minmax.index = range(1, len(X_Part5_minmax)+1)


# %%-----------------------------------------------------------------------
# One Hot Encoding the variables

# encoding categorical features such as gender, education, and marriage status using get dummies
X_Part3 = pd.get_dummies(data.iloc[:, 1:4])
X_Part4 = pd.get_dummies(data.iloc[:, 5:11])

# merge all the parts above
X_Entire = pd.concat([X_Part1_minmax, X_Part2_minmax, X_Part3, X_Part4, X_Part5_minmax], axis=1)

X_data = X_Entire.values
X = X_data[:, :]

# encoding the class with sklearn's LabelEncoder
Y_data = data.values[:, 23]
class_le = LabelEncoder()

# fit and transform the class
y = class_le.fit_transform(Y_data)

# %%-----------------------------------------------------------------------

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# %%----------------------------------------------------------------------- Train
# perform training

# creating the classifier object
clf = SVC(kernel="linear")

# performing training
clf.fit(X_train, y_train)
# %%----------------------------------------------------------------------- Predict

# make predictions

# predict on test

y_predict = clf.predict(X_test)
print(sum(y_predict))

# ----------------------------------------------------------------------- Accuracy

# calculate metrics

print("\n")

print("Classification Report: ")
print(classification_report(y_test,y_predict))
print("\n")

# -----------------------------------------------------------------------
# feature importance, confusion matrix, ROC area mainly used professor's code
# function to display feature importance of the classifier

# display top 20 features (top 10 max positive and negative coefficient values)


def coef_values(coef, names):
    imp = coef
    print(imp)
    imp,names = zip(*sorted(zip(imp.ravel(),names)))

    imp_pos_10 = imp[-10:]
    names_pos_10 = names[-10:]
    imp_neg_10 = imp[:10]
    names_neg_10 = names[:10]

    imp_top_20 = imp_neg_10+imp_pos_10
    names_top_20 = names_neg_10+names_pos_10

    plt.barh(range(len(names_top_20)), imp_top_20, align='center')
    plt.yticks(range(len(names_top_20)), names_top_20)
    plt.show()


# get the column names
features_names = X_Entire.columns.tolist()
# call the function
coef_values(clf.coef_, features_names)

# -----------------------------------------------------------------------

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_predict)
# class_names = data['default payment next month'].unique()
class_names = class_names = ['0', '1']


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )
plt.figure(figsize=(5, 5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                 yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label', fontsize=20)
plt.xlabel('Predicted label', fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------
# Plot ROC Area Under Curve

y_predict_probability = clf.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test,  y_predict_probability)
auc = roc_auc_score(y_test, y_predict_probability)

# print(fpr)
# print(tpr)
# print(auc)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()