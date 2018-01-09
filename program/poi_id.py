# -*- coding:utf-8 -*-
import sys
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pandas as pd
from tester import test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi']  # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# data = featureFormat(data_dict, features)
# 转化为DataFrame格式
data_df = pd.DataFrame(data_dict)
data_df.drop('LOCKHART EUGENE E', axis=1, inplace=True)  # 去除异常值

data_df = data_df.T
data_df = data_df[features_list]
# print data_df.columns
# 每个变量空值个数
for f in features_list:
    if 'NaN' in data_df[f].unique():
        print 'There are %s null values in %s' % (str(data_df[f].value_counts()['NaN']), f)

# 填充空值
data_df.replace({'NaN': 0}, inplace=True)

# 产生新变量
data_df['from_poi_ratio'] = (data_df['from_poi_to_this_person'] / data_df['to_messages']).fillna(0)
data_df['to_poi_ratio'] = (data_df['from_this_person_to_poi'] / data_df['from_messages']).fillna(0)

### Task 2: Remove outliers

# remove the salary outliers
# plt.plot(list(data_df['salary'].values))
data_df = data_df[data_df['salary'] < 2e7]
# plt.plot(list(data_df['salary'].values))
# plt.show()

# remove the total_payments outliers
# plt.plot(list(data_df['total_payments'].values))
data_df = data_df[data_df['total_payments'] < 1e8]
# plt.plot(list(data_df['total_payments'].values))
# plt.show()

# remove the loan_advances outliers
# loan_advances is a bad feature
# plt.plot(list(data_df['loan_advances'].values))
data_df = data_df[data_df['loan_advances'] < 1e8]
# plt.plot(list(data_df['loan_advances'].values))
# plt.show()


# remove the restricted_stock_deferred outliers
# plt.plot(list(data_df['restricted_stock_deferred'].values))
data_df = data_df[data_df['restricted_stock_deferred'] < 1.5e7]
# data_df = data_df[data_df['restricted_stock_deferred'] > -1.75e6]
# plt.plot(list(data_df['restricted_stock_deferred'].values))
# plt.show()



### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
# data = featureFormat(my_dataset, features_list, sort_keys=True)
# labels, features = targetFeatureSplit(data)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

my_features = ['poi', 'salary', 'total_payments', 'bonus', 'deferred_income', 'total_stock_value',
               'expenses', 'exercised_stock_options', 'other',
               'long_term_incentive', 'restricted_stock', 'to_messages', 'from_poi_to_this_person', 'from_messages',
               'from_this_person_to_poi', 'shared_receipt_with_poi', 'from_poi_ratio', 'to_poi_ratio']
labels = data_df[my_features[0]].values
features = data_df[my_features[1:]].values

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

# 根据重要性挑选变量
data_select=SelectKBest(k=10).fit(features,labels)
print my_features[1:]
print data_select.scores_

my_features = ['poi', 'salary', 'total_payments', 'bonus', 'deferred_income', 'total_stock_value',
               'expenses', 'exercised_stock_options',
               'long_term_incentive', 'restricted_stock', 'to_messages', 'from_poi_to_this_person',
               'from_this_person_to_poi', 'shared_receipt_with_poi', 'from_poi_ratio', 'to_poi_ratio']

# 朴素贝叶斯
scaler = StandardScaler()
# pipe_gnb = Pipeline(steps=[('scaler', scaler), ('pca', PCA()), ('clf', GaussianNB())])
pipe_gnb = Pipeline(steps=[('scaler', scaler), ('skb',SelectKBest(k=10)),('pca', PCA(n_components=5)), ('clf', GaussianNB())])
test_classifier(pipe_gnb, labels, features, my_features)

# SVC
# pipe_svc = Pipeline(steps=[('scaler', scaler), ('pca', PCA()), ('clf', SVC())])
pipe_svc = Pipeline(steps=[('scaler', scaler), ('skb',SelectKBest(k=10)),('pca', PCA(n_components=5)), ('clf', SVC())])
test_classifier(pipe_svc, labels, features, my_features)

# 逻辑回归
# pipe_lr = Pipeline(steps=[('scaler', scaler), ('pca', PCA()), ('clf', LogisticRegression())])
pipe_lr = Pipeline(steps=[('scaler', scaler), ('skb',SelectKBest(k=10)),('pca', PCA(n_components=5)), ('clf', LogisticRegression())])
test_classifier(pipe_lr, labels, features, my_features)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.model_selection import train_test_split
#
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectKBest

# pca = PCA(n_components=8)
# selector = SelectKBest(k=5)
# combined_features = FeatureUnion([('pca', pca), ('kbest', selector)])
# combined_features.fit(features, labels).transform(features)
# print 'combined_features:\n', combined_features

# 朴素贝叶斯
print 'GaussianNB:'
pipe_gnb = Pipeline(steps=[('skb', SelectKBest()), ('clf', GaussianNB())])
param_grid = {'skb__k': range(10, 15)}
gs = GridSearchCV(pipe_gnb, param_grid=param_grid)
gs.fit(features, labels)
print gs.best_params_, gs.best_score_

pipe_gnb.set_params(skb__k=gs.best_params_['skb__k'])
test_classifier(pipe_gnb, labels, features, my_features)

# SVC
print 'SVC:'
pipe_svc = Pipeline(steps=[('scaler', scaler), ('pca', PCA()), ('clf', SVC())])
param_grid = {'pca__n_components': range(1,10), 'clf__kernel': ['linear', 'rbf'], 'clf__C': [0.1, 1, 10]}
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
gs = GridSearchCV(pipe_svc, param_grid=param_grid, cv=cv)
gs.fit(features, labels)
print gs.best_params_, gs.best_score_

pipe_svc.set_params(pca__n_components=2)
test_classifier(pipe_svc, labels, features, my_features)

# 逻辑回归
print 'LogisticRegression:'
pipe_lr = Pipeline(steps=[('scaler', scaler), ('pca', PCA()), ('clf', LogisticRegression())])
param_grid = [{'pca__n_components': range(1, 10)}]
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
gs = GridSearchCV(pipe_lr, param_grid=param_grid, cv=cv)
gs.fit(features, labels)

print gs.best_params_, gs.best_score_

pipe_svc.set_params(pca__n_components=6)
test_classifier(pipe_lr, labels, features, my_features)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(pipe_gnb, my_dataset, features_list)
