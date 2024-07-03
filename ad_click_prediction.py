#!/usr/bin/env python
# coding: utf-8

# ### Import libraries. begin, let's import the necessary libraries

# In[66]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[67]:


import warnings
warnings.filterwarnings('ignore')


# In[115]:


get_ipython().system('pip install lightgbm')


# In[116]:


pip install xgboost


# ### Import the dataset

# In[68]:


df = pd.read_csv('Ad Click Data.csv')
df.head()


# In[69]:


print('Number of examples and features:', df.shape)
print('features in the dataset:', df.columns.tolist())


# We have total 1000 training examples and 10 features.

# ### EDA: Exploratory Data Analysis

# In[70]:


df.info()


# In[71]:


# Let's look at stats of the non-object features
df.describe()


# ### Numerical features

# In[72]:


sns.pairplot(df, hue='Clicked on Ad', vars=['Daily Time Spent on Site', 
                                            'Age',
                                           'Area Income',
                                           'Daily Internet Usage'], palette='rocket')


# Pairplot represents the relationshi between the target feature and the explanatory features.
# 
# We also see that users with higher area income who spends more time on the site does not click on ad also relatively younger users with higher income do not click on ads. So this group of users could be the target users.
# 
# Again the users with higher area income who more likely to spend longer time on the site do not click on ad.

# In[73]:


#get the info of the number of ad clicked
fig = plt.figure(figsize = (20, 5))
sns.countplot(x ='Age', data = df)


# We see mjority of the users are in the age range 25 to 45, which could be our target age group for ad recommnedation. We need to check if this age group is actually clicking on the ad or no.

# In[74]:


sns.jointplot(x='Age',y='Daily Time Spent on Site', data=df, hue="Clicked on Ad", palette='rocket')


# In[75]:


sns.scatterplot(x='Age',y='Daily Time Spent on Site', 
                hue='Clicked on Ad', data=df, palette='rocket')


# This plot tells us that the younger users spceially from age 20 to 40, spent most time on the site. So this group of users could be good target group for the ad campaign. We can also say that if a product is targetting a population whose age does not fall into the range 19 to 61, this site is not right platform to advertize the product.
# 
# This plot tells us that all the users who spent less time on the site tend to click on ad. On the other hand, among the 20 to 55 years user group who spent most time on the site apperently don't click on the ad, whereas the same user group who spents less time clicks on ad.

# In[76]:


sns.scatterplot(x='Area Income',y='Daily Time Spent on Site', 
                hue='Clicked on Ad', data=df, palette='rocket')


# This plot tells us that all the users who spent less time on the site and has more area income tend to click on ad. On the other hand, user group with higher area income who spend more time on the site does not seem to click on the ad. Which is interesting, it could be the add is not personalized to this group of users.

# In[77]:


sns.jointplot(x='Age',y='Daily Internet Usage', data=df, hue="Clicked on Ad", palette='rocket')


# Users who spen less time on the internet tends to click on add regardless the age range. On the other hand, users younger than 45 seems to spend more time on the internet but avoid to click on ad.
# 
# 

# In[78]:


sns.jointplot(x='Daily Internet Usage',y='Daily Internet Usage', data=df, hue="Clicked on Ad", palette='rocket')


# We see daily internet use and daily time spent on the site is linearly correlated, which make sense.
# 
# 

# In[79]:


plots = ['Daily Time Spent on Site',
         'Area Income','Daily Internet Usage', 'Age']
for i in plots:
    plt.figure(figsize=(12,6))
    
    plt.subplot(2,3,1)
    sns.boxplot(data=df,x = 'Clicked on Ad', y=i)
    
    plt.subplot(2,3,2)
    sns.boxplot(data=df,y=i)
    
    plt.subplot(2,3,3)
    sns.distplot(df[i],bins=20)
    plt.tight_layout()
    plt.title(i)
    plt.show()


# In[80]:


df_final = df.select_dtypes(include=np.number)
df_final.head()


# In[81]:


df_final.corr()


# In[82]:


fig = plt.figure(figsize=(10,8))
sns.heatmap(df_final.corr(), annot=True)


# Here we see Daily Time Spent on Site, Age, Area Income, Daily Intenert Usage are highly correlate with the target variable. Which indicates they are important features and will be useful for ML model.
# 
# We also notice that Daily Time spent on site has strong correlation with other with daily intenet usage, Age, Area Inocme and so does daily internet usage

# ### Categorical Features

# In[83]:


fig = plt.figure(figsize = (6,2))
sns.countplot(data = df , y = 'Male')
print(df['Male'].value_counts())


# In[84]:


#get the info of the number of ad clicked
fig = plt.figure(figsize = (6,2))
sns.countplot(y ='Clicked on Ad', data = df)
print(df['Clicked on Ad'].value_counts())


# Data seems very balanced interms of Male feature and the target feature.
# 
# 

# In[85]:


#get the info of the Ad Topic Line
print(df['Ad Topic Line'].value_counts())


# In[86]:


object_features = ['Ad Topic Line', 'City', 'Country', 'Timestamp']
df[object_features].describe(include=['O'])


# From above cell we see that all ad topic lines are unique, which indicates this features has less chace of carying any useful information for the prediction model. There are 969 diffirent cities out of 237 countries. These indicates that the users are not from a spcecific demograhic but from all over the world. Even though we see France repeates 9 times, meaning highest number of visitors are from France but still it just 9 of them.

# In[87]:


df['Timestamp'] = pd.to_datetime(df['Timestamp'])

df['Month'] = df['Timestamp'].dt.month
df['Day'] = df['Timestamp'].dt.day
df['Weekday'] = df['Timestamp'].dt.dayofweek
df['Hour'] = df['Timestamp'].dt.hour
df = df.drop(['Timestamp'], axis=1)

df.head()


# In[88]:


df['Month'].unique()


# In[89]:


df['Month'][df['Clicked on Ad'] == 1].value_counts().sort_index()


# In[90]:


df['Month'][df['Clicked on Ad'] == 1].value_counts().sort_index().plot()


# In[91]:


df['Day'][df['Clicked on Ad'] == 1].value_counts().sort_index().plot()


# In[92]:


df['Weekday'][df['Clicked on Ad'] == 1].value_counts().sort_index()


# In[93]:


df['Weekday'][df['Clicked on Ad'] == 1].value_counts().sort_index().plot()


# In[94]:


df['Hour'][df['Clicked on Ad'] == 1].value_counts().sort_index().plot()


#  ## Logistic Regression

# #### Missing Values

# In[95]:


missing = df.isnull().sum()
missing_precent = 100*missing/len(df)
missing_table = pd.concat([missing, missing_precent], axis=1)
missing_table.columns = ['missing_value', '% of missing_value']
missing_table = missing_table.loc[missing_table['missing_value'] != 0].sort_values('missing_value')
print('The dataset has total {} columns \nThere are {} columns that have missing values\n\n'.format(df.shape[1], missing_table.shape[0]))
missing_table.head()


# In[96]:


ncounts = pd.DataFrame([df.isna().mean()]).T
ncounts = ncounts.rename(columns={0: "train_missing", 1: "test_missing"})

ncounts.query("train_missing > 0").plot(
    kind="barh", figsize=(8, 5), title="% of Values Missing"
)
plt.show()


# In[97]:


df.select_dtypes(exclude='object').shape


# In[98]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

#filling null values with the median values
imputer = SimpleImputer()
#using features with only neumeric values 
#since we already decided to drop all the object features in this dataset
df_numeric = df.select_dtypes(exclude='object')
imputed_df = imputer.fit_transform(df_numeric)

# since imputation converts the data frame into a numpy array, 
# let's convert it into a dataframe back
df_train = pd.DataFrame(imputed_df)

# since imputation removes column names, let's put them back
df_train.columns = df.select_dtypes(exclude='object').columns
train_features = ['Daily Time Spent on Site', 'Age', 'Area Income',
                   'Daily Internet Usage', 'Male', 
                   'Month', 'Day', 'Weekday', 'Hour']

numeric_features = ['Daily Time Spent on Site', 'Age', 'Area Income',
                   'Daily Internet Usage']

scaler = StandardScaler()
df_train[numeric_features] = scaler.fit_transform(df_train[numeric_features])

X = df_train[train_features]
y = df_train['Clicked on Ad']

x_train, x_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25,
                                                    random_state=101)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# In[99]:


x_train.head()


# In[100]:


x_train.isnull().sum().sum()


# In[101]:


LR = LogisticRegression(solver='lbfgs')
LR.fit(x_train, y_train)
predictions_LR = LR.predict(x_test)

print('\nLogistic regression accuracy:', accuracy_score(predictions_LR, y_test))

cf_matrix = confusion_matrix(predictions_LR, y_test)
print('\nConfusion Matrix:')
print(cf_matrix)

print(classification_report(y_test, predictions_LR))


# In[102]:


from sklearn import metrics



cf_matrix = metrics.confusion_matrix(y_test, predictions_LR)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cf_matrix, display_labels = [0, 1])

cm_display.plot()
plt.show()


#      Confusion Matrix: The users that are predicted to click on commercials and the actually clicked were 112, the people who were predicted not to click on the commercials and actually did not click on them were 129.
# 
# The people who were predicted to click on commercial and actually did not click on them are 5, and the users who were not predicted to click on the commercials and actually clicked on them are 2.
# 
# We have only a few mislabelled points which is not bad from the given size of the dataset.
# 
# Classification Report:
# 
# From the report obtained, the precision & recall are 0.96 which depicts the predicted values are 98% accurate. Hence the probability that the user can click on the commercial is 0.96 which is a great precision value to get a good model.

# ## Decision Tree Classifier

# In[103]:


from sklearn.tree import DecisionTreeClassifier

model2 = DecisionTreeClassifier()
model2.fit(x_train, y_train)
predictions_DT = model2.predict(x_test)

print('\nDecisionTreeClassifier accuracy:', accuracy_score(predictions_DT, y_test))
cf_matrix = confusion_matrix(predictions_DT, y_test)
print('\nConfusion Matrix:')
print(cf_matrix)


# In[104]:


from sklearn import metrics



cf_matrix = metrics.confusion_matrix(y_test, predictions_DT)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cf_matrix, display_labels = [0, 1])

cm_display.plot()
plt.show()


# ## Naive Bayes

# In[105]:


from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(x_train, y_train)


# In[106]:


from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)

y_pred2 = model.predict(x_test)

accuray = accuracy_score(y_pred2, y_test)
f1 = f1_score(y_pred2, y_test, average="weighted")

print("Accuracy:", accuray)
print("F1 Score:", f1)


# In[107]:


from sklearn import metrics



cf_matrix = metrics.confusion_matrix(y_test, y_pred2)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cf_matrix, display_labels = [0, 1])

cm_display.plot()
plt.show()


# In[108]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred2))


# ## KNeighborsClassifier

# In[109]:


from sklearn.neighbors import KNeighborsClassifier

k_values = [i for i in range (1,31)]
scores = []



for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    scores.append(accuracy_score(y_test,knn.predict(x_test)))


# In[110]:


sns.lineplot(x = range(1,31),y = scores, marker = 'o')


# In[111]:


best_index = np.argmax(scores)
best_k = k_values[best_index]

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train, y_train)


# In[112]:


y_pred = knn.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# ## XGBoost Classifier

# In[117]:


from xgboost import XGBClassifier

model3 = XGBClassifier()
model3.fit(x_train, y_train)
predictions_XGB = model3.predict(x_test)

print('\nXGBClassifier accuracy:', accuracy_score(predictions_XGB, y_test))
cf_matrix = confusion_matrix(predictions_DT, y_test)
print('\nConfusion Matrix:')
print(cf_matrix)


# ### Feature Importance

# In[121]:


import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

feature_importances = np.zeros(X.shape[1])

# Create the model with several hyperparameters
model = lgb.LGBMClassifier(objective='binary',
                           boosting_type = 'goss',
                           n_estimators = 10000,
                           class_weight = 'balanced')

# Fit the model twice to avoid overfitting
for i in range(2):

    # Split into training and validation set
    train_features, valid_features, train_y, valid_y = train_test_split(X,
                                                                        y,
                                                                        test_size = 0.25,
                                                                        random_state = i)

    # Define the early stopping callback
    callback = lgb.early_stopping(stopping_rounds=100, verbose=200)

    # Train using early stopping
    model.fit(train_features, train_y, eval_set = [(valid_features, valid_y)],
              eval_metric = 'auc', callbacks=[callback])
    predictions_LGB = model.predict(valid_features)

    print('\nLGB accuracy:', accuracy_score(predictions_LGB, valid_y))
    print('\nConfusion Matrix:')
    print(confusion_matrix(predictions_LGB, valid_y))

    # Record the feature importances
    feature_importances += model.feature_importances_


# In[123]:


# Make sure to average feature importances! 
feature_importances = feature_importances / 2
feature_importances = pd.DataFrame({'feature': list(X.columns),
                                    'importance': feature_importances}
                                  ).sort_values('importance', ascending = False)

feature_importances.head(10)


# In[124]:


# Find the features with zero importance
zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
print('There are %d features with 0.0 importance' % len(zero_features))
feature_importances.tail()


# In[125]:


def plot_feature_importances(df, threshold = 0.9):
    """
    Plots 10 most important features and the cumulative importance of features.
    Prints the number of features needed to reach threshold cumulative importance.
    
    Parameters
    --------
    df : dataframe
        Dataframe of feature importances. Columns must be feature and importance
    threshold : float, default = 0.9
        Threshold for prining information about cumulative importances
        
    Return
    --------
    df : dataframe
        Dataframe ordered by feature importances with a normalized column (sums to 1)
        and a cumulative importance column
    
    """
    
    plt.rcParams['font.size'] = 18
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    # Cumulative importance plot
    plt.figure(figsize = (8, 6))
    plt.plot(list(range(len(df))), df['cumulative_importance'], 'r-')
    plt.xlabel('Number of Features'); plt.ylabel('Cumulative Importance'); 
    plt.title('Cumulative Feature Importance');
    plt.show();
    
    importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
    print('%d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold))
    
    return df


# In[126]:


norm_feature_importances = plot_feature_importances(feature_importances,
                                                   threshold = 0.99)


# In[ ]:




