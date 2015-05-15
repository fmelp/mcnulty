import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import seaborn as sns
import pydot
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.learning_curve import learning_curve
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

# class Logger(object):
#     def __init__(self, filename="Default.log"):
#         self.terminal = sys.stdout
#         self.log = open(filename, "w")
#sys.stdout = open('stdout.txt', 'w')

#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)



def split_data(data):
    """
    @param -> data : pandas DataFrame
    @return -> train, test : split into 2 pandas DFs with NO HEADERS
    """
    train, test = train_test_split(data)
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    return train, test


def train_clf(clf, train_X, train_y, test_X, test_y):
    '''
    @param -> clf : classifier you want to use, must be set up before
              train_X,train_y,test_X,test_y : data split into train and test
                                               and X and y
    @return -> accuracy : accuracy of model (correct pred / len(test_y))
               result : np array of binary classification results
               result_prob : np array of classification results with probabilities

    '''
    clf.fit(train_X, train_y)
    result_prob = clf.predict_proba(test_X)
    result = clf.predict(test_X)
    accuracy = accuracy_score(test_y, result)
    return accuracy, result, result_prob


def plot_roc(test_y, result_prob):
    '''
    @param -> test_y : the original y (actual values for the pred)
              result_prob : classification results with probabilities
    @return -> NULL : only displays a graph
    '''
    false_positive_rate, recall, thresholds = roc_curve(test_y, result_prob[:,1])
    roc_auc = auc(false_positive_rate, recall)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Recall')
    plt.xlabel('Fall-out')
    plt.show()

def create_graph(clf, filename):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(filename)


census_data = pd.read_csv("~/Desktop/metis/mcnulty_project/census_data.csv",
                          header=None)
# census_data[census_data=='?'] = np.nan
census_data.replace(to_replace='?', value=np.nan, inplace=True)

#change y(to_pred) to binary
census_data[14] = census_data[14].apply(lambda x: 1 if x == ' >50K' else 0)

# change certain columns to categorical
cols_to_categorical = [1, 3, 5, 6,
                       7, 8, 9, 13]
for col in cols_to_categorical:
    census_data[col] = census_data[col].astype('category')
    census_data = pd.concat([census_data, pd.get_dummies(census_data[col])], axis=1)


# split data to train and test
train, test = split_data(census_data)


train_X = train.drop([1, 3, 5, 6, 7, 8, 9, 13, 14], axis=1)
test_X = test.drop([1, 3, 5, 6, 7, 8, 9, 13, 14], axis=1)
train_y = train[14].astype('category')
test_y = test[14].astype('category')

# decision tree
# dt_clf = DecisionTreeClassifier(max_depth=5)
# dt_accuracy, dt_result, dt_result_prob = train_clf(dt_clf,
#     train_X, train_y, test_X, test_y)

# dt_clf = DecisionTreeClassifier()

# grid_search_dt = GridSearchCV(dt_clf, param_grid, scoring='accuracy', cv=5, verbose=10)
# print grid_search_dt.best_params_
# print grid_search_dt.best_estimator_
# grid_search_dt.fit(train_X, train_y)
# gs_result_prob = grid_search_dt


def get_best_model_params(clf, train_X, train_y, param_grid,
                           scoring_metric, cv):
    '''
    @param -> train_X : n-feature matrix : train feature data
              train_y : 1-feature matrix : train result data
              clf : sklearn_classifier : simply initiated classifier of choice
              param_grid : list of dictionaries :
                           [{'max_depth':[1,2,3]}] : of parameters to tweak
              scoring_metric : str : accuracy, precision, recall, f1 or others(?)
              cv : int : number of times to run cross cross validation

    @return -> best_estimator_ : sklearn_classifier : classifier tuned w best params
               grid_scores : list : summary of results

    NOTE: Look @ output in console to see runtimes of each of the params

    '''
    #can add verbose=10 for logging
    #try and get this to store the seconds by redirecting output to file, then getting the info
    #on params: avg_score, avg_time
    #sys.stdout = Logger("log.txt")
    # with open('help.txt', 'w') as f:
    #     with redirect_stdout(f):
    #         print('it now prints to `help.text`')
    #sys.stdout = open('stdout.txt', 'w')
    grid_search = GridSearchCV(clf, param_grid,
                                   scoring=scoring_metric, cv=cv, verbose=10)
    grid_search.fit(train_X, train_y)
    #plot the time to metric ratio
    # if plot_time_metric:
    # with open("log.txt", "r") as f:
    #     l = f.readlines()
    #     l = [x for x in l if x[len(x)-1] == 's']


    return grid_search.best_estimator_, grid_search.grid_scores_

param_grid = [{'max_depth' : [5, 3, 2, 13, 12, 20]}]
dt_clf, dt_clf_scores = get_best_model_params(DecisionTreeClassifier(), train_X, train_y, param_grid, 'recall', 5)

print dt_clf

param_grid_svc = [{"C": [0.01, 0.1, 1], "kernel": ['rbf'], "gamma":[0.0, 0.01, 0.1, 1]}]
svc_clf, svc_clf_scores = get_best_model_params(SVC(), train_X, train_y, param_grid_svc, 'recall', 3)

print svc_clf
print svc_clf_scores

#learning curve -> to give you optimal trainset size

#reads the file
# with open("log.txt", "r") as f:
#         l = f.readlines()
#         # l = [x for x in l if x[len(x)-1] == 's' and '|' not in x]
#         print l


# logistic regression
# logit_clf = LogisticRegression()
# logit_clf.fit(train_X, train_y)
# lg_result_prob = logit_clf.predict_proba(test_X)
# lg_result = logit_clf.predict(test_X)
# lg_accuracy = accuracy_score(test_y, lg_result)
# lg_accuracy, lg_result, lg_result_prob = train_clf(logit_clf,
#     train_X, train_y, test_X, test_y)

# SVM

print cross_val_score(dt_clf, census_data.drop([1, 3, 5, 6, 7, 8, 9, 13, 14], axis=1), census_data[14], cv=10, verbose=10, scoring='recall')

#SVM:
# svm_clf, svm_clf_scores = get_best_model_params(SVC(), train_X, train_y, param_grid, 'recall', 5)
# print cross_val_score(svm_clf, census_data.drop([1, 3, 5, 6, 7, 8, 9, 13, 14], axis=1), census_data[14], cv=10, verbose=10, scoring='recall')

create_graph(dt_clf, 'tree2.pdf')
#plot_roc(test_y, dt_result_prob)
#plot_roc(test_y, lg_result_prob)

