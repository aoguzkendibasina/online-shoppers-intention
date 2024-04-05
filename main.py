###########################################################
# End-to-End Purchasing Intention Machine Learning Pipeline
###########################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Base Models
# 4. Automated Hyperparameter Optimization
# 5. Prediction for a New Observation

# Importing requirement Python libraries:
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Setting options for display:
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)

# Disabling Python warnings:
import warnings
warnings.simplefilter(action='ignore', category=Warning)

################################################
# 1. Exploratory Data Analysis
################################################

# Importing utils file:
import utils

# Loading the dataset and checking the data frame:
df = pd.read_csv("online_shoppers_intention.csv")
utils.check_df(df)

# Decomposition of numeric and categorical variables:
cat_cols, num_cols, cat_but_car = utils.grab_col_names(df, cat_th=14, car_th=20)
num_cols.remove('TrafficType')
cat_cols.append('TrafficType')
cat_cols.remove('SpecialDay')
num_cols.append('SpecialDay')

# Examination of categorical variables:
for col in cat_cols:
    utils.cat_summary(df, col)

# Examination of numeric variables:
df[num_cols].describe().T

for col in num_cols:
     utils.num_summary(df, col, plot=True)

# Correlation of numeric variables with each other:
utils.correlation_matrix(df, num_cols)

# Examination of numeric variables with target variable:
for col in num_cols:
    utils.target_summary_with_num(df, "Revenue", col)

# Examination of categorical variables with target variable:
for col in cat_cols:
    utils.target_summary_with_cat(df, "Revenue", col)

################################################
# 2. Data Preprocessing & Feature Engineering
################################################

# Feature Extraction
df["Season"] = df["Month"]
df["Season"].replace(["Dec", "Feb"], "Winter", inplace=True)
df["Season"].replace(["Mar", "May"], "Spring", inplace=True)
df["Season"].replace(["June", "Jul", "Aug"], "Summer", inplace=True)
df["Season"].replace(["Sep", "Oct", "Nov"], "Autumn", inplace=True)
cat_cols.append("Season")

df['Administrative_Duration_pervisit'] = df['Administrative_Duration'] / df['Administrative']
df['Informational_Duration_pervisit'] = df['Informational_Duration'] / df['Informational']
df['ProductRelated_Duration_pervisit'] = df['ProductRelated_Duration'] / df['ProductRelated']
df = df.fillna(0)

num_cols.append('Administrative_Duration_pervisit')
num_cols.append('Informational_Duration_pervisit')
num_cols.append('ProductRelated_Duration_pervisit')

# Removing the target variable from the categorical column list:
cat_cols = [col for col in cat_cols if "Revenue" not in col]

# The number of numerical columns = 13
len(num_cols)
# The number of categorical columns = 9
len(cat_cols)

# One-hot encoding:
df = utils.one_hot_encoder(df, cat_cols, drop_first=True)

# The number of columns = 75
len(df.columns)

# Checking outliers:
for col in num_cols:
    print(col, utils.check_outlier(df, col, 0.05, 0.95))

# Standardization of numerical variables:
scaler = RobustScaler()
scaler.fit(df[num_cols])

# Saving robust scaler:
joblib.dump(scaler, "scaler.save")

X_scaled = scaler.transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

# Dependant variable:
y = df["Revenue"]

# Independant variables:
X = df.drop(["Revenue"], axis=1)

######################################################
# Feature Importances
######################################################

# Splitting the dataset into random train and test subsets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

# Modelling:
rf_model = RandomForestClassifier(random_state=45).fit(X_train, y_train)

# Prediction:
y_pred = rf_model.predict(X_test)

# Scoring:
accuracy_score(y_test, y_pred)
# 0.9045688023790214

f1_score(y_test, y_pred)
# 0.6430738119312437

roc_auc_score(y_test, y_pred)
# 0.7634811584293368

print(classification_report(y_test, y_pred))

confusion_matrix(y_test, y_pred)

# Plotting confusion matrix:
utils.plot_confusion_matrix(y_test, y_pred)

# Plotting feature importances:

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train)


######################################################
# 3. Base Models
######################################################

utils.base_models(X, y)
utils.base_models(X, y, scoring="f1")
utils.base_models(X, y, scoring="accuracy")

'''
Base Models....
roc_auc: 0.8529 (LR) 
roc_auc: 0.5959 (CART) 
roc_auc: 0.8076 (RF) 
roc_auc: 0.77 (Adaboost) 
roc_auc: 0.7688 (GBM) 
roc_auc: 0.728 (XGBoost) 
roc_auc: 0.6824 (LightGBM) 
'''

'''
Base Models....
f1: 0.4563 (LR) 
f1: 0.3368 (CART) 
f1: 0.3328 (RF) 
f1: 0.2745 (Adaboost) 
f1: 0.2747 (GBM) 
f1: 0.3666 (XGBoost) 
f1: 0.3581 (LightGBM) 
'''

'''
Base Models....
accuracy: 0.8735 (LR) 
accuracy: 0.6135 (CART) 
accuracy: 0.7295 (RF) 
accuracy: 0.6751 (Adaboost) 
accuracy: 0.6496 (GBM) 
accuracy: 0.6671 (XGBoost) 
accuracy: 0.6546 (LightGBM) 
'''

######################################################
# 4. Automated Hyperparameter Optimization
######################################################

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20, 25],
             "n_estimators": [200, 300]}

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}

xgboost_params = {"learning_rate": [0.01, 0.1, 0.3],
                  "max_depth": [4, 5, 6, 8],
                  "n_estimators": [100, 150, 200]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 400, 500]}

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

adaboost_params = {"learning_rate": [0.001, 0.01, 0.1],
                   "n_estimators": [100, 200]}

classifiers = [("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('GBM', GradientBoostingClassifier(), gbm_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(verbosity=-1), lightgbm_params),
               ('AdaBoost', AdaBoostClassifier(), adaboost_params)]

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y)
best_models = hyperparameter_optimization(X, y, scoring="f1")
best_models = hyperparameter_optimization(X, y, scoring="accuracy")


###############################################################
# Coping with the unbalanced data set with Undersampling
###############################################################

# Number of classes in the dataset before random undersampling:
y.value_counts()
'''
Revenue
False    10422
True      1908
'''

# Importing the library:
from imblearn.under_sampling import RandomUnderSampler

# Transforming the dataset:
ranUnSample = RandomUnderSampler()
X_ranUnSample, y_ranUnSample = ranUnSample.fit_resample(X, y)

# Number of classes in the dataset after random undersampling:
y_ranUnSample.value_counts()
'''
Revenue
False    1908
True     1908
'''

# Modelling after undersampling:

# Base Models

utils.base_models(X_ranUnSample, y_ranUnSample)
utils.base_models(X_ranUnSample, y_ranUnSample, scoring="f1")
utils.base_models(X_ranUnSample, y_ranUnSample, scoring="accuracy")

'''
Base Models....
f1: 0.61 (LR) 
f1: 0.5225 (CART) 
f1: 0.5644 (RF) 
f1: 0.5672 (Adaboost) 
f1: 0.5646 (GBM) 
f1: 0.558 (XGBoost) 
f1: 0.5652 (LightGBM) 
'''

# Automated Hyperparameter Optimization
best_models = hyperparameter_optimization(X_ranUnSample, y_ranUnSample)
best_models = hyperparameter_optimization(X_ranUnSample, y_ranUnSample, scoring="f1")
best_models = hyperparameter_optimization(X_ranUnSample, y_ranUnSample, scoring="accuracy")

'''

Hyperparameter Optimization....
########## CART ##########
f1 (Before): 0.5249
f1 (After): 0.8257
CART best params: {'max_depth': 1, 'min_samples_split': 2}

########## RF ##########
f1 (Before): 0.5624
f1 (After): 0.566
RF best params: {'max_depth': 8, 'max_features': 'auto', 'min_samples_split': 15, 'n_estimators': 300}

########## GBM ##########
f1 (Before): 0.5644
f1 (After): 0.5687
GBM best params: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.5}

########## XGBoost ##########
f1 (Before): 0.558
f1 (After): 0.5621
XGBoost best params: {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 200}

########## LightGBM ##########
f1 (Before): 0.5652
f1 (After): 0.5637
LightGBM best params: {'learning_rate': 0.01, 'n_estimators': 300}

########## AdaBoost ##########
f1 (Before): 0.5672
f1 (After): 0.8257
AdaBoost best params: {'learning_rate': 0.001, 'n_estimators': 100}

'''

# F1 Score --> CART: 0.8257 and AdaBoost: 0.8257

# Re-Hyperparameter Optimization for CART ve AdaBoost:

cart_params = {'max_depth': range(1, 30),
               "min_samples_split": range(1, 50)}


adaboost_params = {"learning_rate": [0.0001, 0.001, 0.01, 0.1],
                   "n_estimators": [75, 90, 100, 110, 125, 150, 175, 200]}

classifiers = [("CART", DecisionTreeClassifier(), cart_params),
               ('AdaBoost', AdaBoostClassifier(), adaboost_params)]

best_models = hyperparameter_optimization(X_ranUnSample, y_ranUnSample, scoring="f1")

# F1 Score --> CART: 0.8257 and AdaBoost: 0.8257

######################################################################
# Hyperparameter Optimization with RandomSearchCV for Adaboost Model
######################################################################

adaboost_model = AdaBoostClassifier(random_state=17)

adaboost_random_params = {"learning_rate": np.random.uniform(0, 0.1, 10),
                          "n_estimators": [int(x) for x in np.linspace(start=50, stop=200, num=10)]}

adaboost_random = RandomizedSearchCV(estimator=adaboost_model,
                               param_distributions=adaboost_random_params,
                               n_iter=100,
                               cv=3,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)

adaboost_random.fit(X_ranUnSample, y_ranUnSample)


adaboost_random.best_params_
# {'n_estimators': 50, 'learning_rate': 0.03508745532730652}

adaboost_random_final = adaboost_model.set_params(**adaboost_random.best_params_, random_state=17).fit(X_ranUnSample, y_ranUnSample)

cv_results = cross_validate(adaboost_random_final, X_ranUnSample, y_ranUnSample, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.8482464507009393
cv_results['test_f1'].mean()
# 0.8364201593405415
cv_results['test_roc_auc'].mean()
# 0.8461198080962721

######################################################
# 5. Prediction for a New Observation
######################################################

# Choosing a random user:
random_user = X.sample(1, random_state=45)

# Prediction for the random user:
adaboost_random_final.predict(random_user)

# Saving the final model:
joblib.dump(adaboost_random_final, "adaboost.pkl")

# Loading the model and making prediction:
new_model = joblib.load("adaboost.pkl")
new_model.predict(random_user)


