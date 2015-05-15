import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.learning_curve import learning_curve
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as sm2
import statsmodels.api as sm
import re


def split_data(data):
    """
    @param -> data : pandas DataFrame
    @return -> train, test : split into 2 pandas DFs with NO HEADERS
    """
    column_names = list(data.columns.values)
    column_dict = dict(zip(range(len(column_names)), column_names))
    train, test = train_test_split(data)
    train = pd.DataFrame(train)
    train.rename(columns=column_dict, inplace=True)
    test = pd.DataFrame(test)
    test.rename(columns=column_dict, inplace=True)
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


def get_client_zips():
    '''
    call function to return a dictionary of previous clients
        key: zipcode  <---> value: number of guests from that zipcode
    '''
    bd = pd.read_csv("~/Desktop/mcnulty_project/borgo_data/Ospiti.csv", sep=';')
    bd['CAP'] = bd['CAP'].dropna().apply(lambda x: re.sub("[^0-9]", "", x)).apply(lambda x: x if len(x) == 5 else np.nan)
    bd = bd[bd['CAP'] != np.nan]
    bd = bd[~(bd['CAP'].isnull())]

    zip_list = bd['CAP'].tolist()
    zip_dict = dict( [ (i, zip_list.count(i)) for i in set(zip_list) ] )
    return zip_dict


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
    grid_search = GridSearchCV(clf, param_grid,
                                   scoring=scoring_metric, cv=cv, verbose=10)
    grid_search.fit(train_X, train_y)
    return grid_search.best_estimator_, grid_search.grid_scores_



#read data
city_data = pd.read_csv("~/Desktop/city_data.csv")

#append a column full of nan
num_guests = [np.nan for x in range(len(city_data))]
city_data['num_guests'] = num_guests

# loops through to give num_guests column a value
zip_dict = get_client_zips()
for i, row in city_data.iterrows():
    if list(str(row['zip']))[0] == '1' and row['zip'] in np.array(zip_dict.keys(), dtype=np.float):
        row['num_guests'] = zip_dict[''.join(list(str(row['zip']))[:-2])]
        city_data.loc[i:i, 'num_guests'] = zip_dict[''.join(list(str(row['zip']))[:-2])]
    elif list(str(row['zip']))[0] == '1' and str(row['zip']) not in zip_dict.keys():
        city_data.loc[i:i, 'num_guests'] = 0
    else:
        city_data.loc[i:i, 'num_guests'] = 0



#get only ny zips
ny = city_data[(city_data['zip'] < 60000)].dropna()
#split it into train and test
train_ny, test_ny = split_data(ny)
#split into X, y
train_ny_X = train_ny.drop(['num_guests', 'zip'], 1)
train_ny_y = train_ny['num_guests']
test_ny_X = test_ny.drop(['num_guests', 'zip'], 1)
test_ny_y = test_ny['num_guests']
# scale data
std_scale = StandardScaler().fit(train_ny_X)
train_ny_X_std = std_scale.transform(train_ny_X)
test_ny_X_std = std_scale.transform(test_ny_X)
#run Poisson GLM
poisson_glm = sm.GLM(train_ny_y, train_ny_X_std, family=sm.families.Poisson())
poisson_results = poisson_glm.fit()
print poisson_results.summary()
ny_test_res = poisson_results.predict(test_ny_X_std)
#score results
scores = []
for i in xrange(len(ny_test_res)):
    if test_ny_y.tolist()[i] != 0:
        scores.append((test_ny_y.tolist()[i] - ny_test_res[i])/test_ny_y.tolist()[i])
    else:
        scores.append(test_ny_y.tolist()[i] - ny_test_res[i])
print scores


#all ny together + scale
ny_X = ny.drop(['num_guests', 'zip'], 1)
ny_y = ny['num_guests']
std_scale2 = StandardScaler().fit(ny_X)
ny_X_std = std_scale2.transform(ny_X)

#get only chicago zips
chicago = city_data[~(city_data['zip'] < 60000)].dropna()
#split into X, y
chicago_X = chicago.drop(['num_guests', 'zip'], 1)
chicago_y = chicago['num_guests']
chicago_X_std = std_scale2.transform(chicago_X)

poisson_glm2 = sm.GLM(ny_y, ny_X_std, family=sm.families.Poisson())
poisson_results = poisson_glm.fit()
print poisson_results.summary()
ny_ch_res = poisson_results.predict(chicago_X_std)
print np.array(ny_ch_res).median
print sorted(ny_ch_res)[len(ny_ch_res)//2]
percentages = [round((x*100/sorted(ny_ch_res)[len(ny_ch_res)//2])-100,1) for x in ny_ch_res]
print len(percentages)
ch_zps = chicago['zip'].tolist()
ch_zps = [str(x) for x in ch_zps]
print ch_zps
print dict(zip(ch_zps, percentages))


