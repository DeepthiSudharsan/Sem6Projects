from metaheuristic_custom.firefly_algorithm import FireflyAlgorithm
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sys
from metaheuristic_custom.function_wrappers.abstract_wrapper import AbstractWrapper
import numpy as np
from lssvr import LSSVR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sys

df = pd.read_csv("GOOG.csv", keep_default_na = False)
df["close_prev"] = df["Close"].shift(1) # Shifting the closing prices by 1 for easier calculation
df["return"] = np.log(df["Close"])-np.log(df["close_prev"]) # Return: diff of the log values
df = df.iloc[1: , :] # Neglecting 1st value as it becomes NaN
train_len = round(80*len(df['Date'])/100)
test_len = len(df['Date']) - train_len
train_date = list(df['Date'])[train_len]
print(train_date)
test_date = '2021-04-08'
td = datetime.strptime(test_date,'%Y-%m-%d')
dates = []
for i in range(len(df)):
    dates.append(datetime.strptime(df.iloc[i].Date,'%Y-%m-%d'))

df['date_f'] = dates
train_df = df[df.date_f < td]
test_df = df[df.date_f >= td]
train_data = train_df['Close']
test_data = test_df['Close']
plt.figure(figsize=(20,5))
plt.plot(train_data)
plt.plot(test_data)
plt.show()
scaler = MinMaxScaler()
train_data = train_data.values.reshape(-1, 1)
test_data = test_data.values.reshape(-1, 1)
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)
timesteps = 25
train_data_timesteps = np.array([[j for j in train_data[i:i+timesteps]] for i in range(0, len(train_data)-timesteps+1)])[:, :, 0]
print(train_data_timesteps.shape)
test_data_timesteps = np.array([[j for j in test_data[i:i+timesteps]] for i in range(0, len(test_data)-timesteps+1)])[:, :, 0]
print(test_data_timesteps.shape)

x_train, y_train = train_data_timesteps[:, :timesteps-1], train_data_timesteps[:, [timesteps-1]]
x_test, y_test = test_data_timesteps[:, :timesteps-1], test_data_timesteps[:, [timesteps-1]]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

model = LSSVR(kernel='rbf', gamma=0.002, C=50)
model.fit(x_train, y_train)
y_hat = model.predict(x_test)
print('MSE', mean_squared_error(y_test, y_hat))
print('R2 Score',model.score(x_test, y_test))

plt.plot(y_test)
plt.plot(y_hat)
plt.show()


X= np.array(x_train)
X_test=np.array(x_test)
y=np.array(y_train)
y_test=np.array(y_test)

# 1. INIT DATASET==================================================


# 2. ALL FUNC DEFINITION============================================
def Train(subset_X,subset_y,pars):
    stock_clf = Pipeline([
        ('clf', LSSVR(kernel='rbf', C=pars[0], gamma=pars[1])),
    ])
    clf = stock_clf.fit(subset_X, subset_y)
    return clf



def Testing(clf, subset_X, subset_y):
    expected = subset_y
    predicted = clf.predict(subset_X)
    acc = mean_squared_error(expected, predicted)
    return acc


def SearchingParameters(train_X, train_y, test_X, test_y, pars):
    cl = Train(train_X, train_y, pars)
    # print ("Parameters : "+str(pars))
    return Testing(cl, test_X, test_y)


number_of_variables = 2
objective = "minimization"

# firefly_algorithm 
number_of_fireflies = 10
maximum_generation = 10
randomization_parameter_alpha = 0.2
absorption_coefficient_gamma = 1.0

min_pars = [50, 0.001]
max_pars = [200, 0.05]
initial_pars = [50, 0.002]

class SearchSVMParameter(AbstractWrapper):

    def __init__(self, subset_X_train, subset_y_train, subset_X_test, subset_y_test):
        self.subset_X_train = subset_X_train
        self.subset_y_train = subset_y_train

        self.subset_X_test = subset_X_test
        self.subset_y_test = subset_y_test
        self.REPORT = ""
        self.LOGGING = []

    def maximum_decision_variable_values(self):
        return max_pars

    def minimum_decision_variable_values(self):
        return min_pars

    def objective_function_value(self, decision_variable_values):
        return SearchingParameters(self.subset_X_train, self.subset_y_train, self.subset_X_test, self.subset_y_test,
                                   decision_variable_values)

    def initial_decision_variable_value_estimates(self):
        return initial_pars

    def logging(self, datas):
        for k, v in datas.items():
            current = str(k) + " : " + str(v) + "\n"
            self.REPORT += current

        self.REPORT += "\n\n"
        self.LOGGING.append(datas)

    def report(self):
        return self.REPORT
# 3. START SEARCHING==================================================

def SearchFireFly(subset_X_train, subset_y_train, subset_X_test, subset_y_test):
    fc = SearchSVMParameter(subset_X_train, subset_y_train, subset_X_test, subset_y_test)

    firefly_algorithm = FireflyAlgorithm(fc, number_of_variables, objective)

    result = firefly_algorithm.search(
        number_of_fireflies=number_of_fireflies,
        maximum_generation=maximum_generation,
        randomization_parameter_alpha=randomization_parameter_alpha,
        absorption_coefficient_gamma=absorption_coefficient_gamma)

    print("Best parameters : " + str(result["best_decision_variable_values"]))
    print("Best MSE : " + str(result["best_objective_function_value"]))
    return result


def KFOLDS(kf):
    skf = KFold(n_splits=kf)
    folds = skf.split(X, y)
    fold = 1
    REPORT = ""
    best_gamma = float("inf")
    best_c = float("inf")
    best_acc = float("inf")

    for train_index, test_index in folds:
        print("\n\nFOLDS=" + str(fold))
        training_X = []
        training_y = []
        testing_X = []
        testing_y = []

        for i in train_index:
            training_X.append(X[i])
            training_y.append(y[i])

        for i in test_index:
            testing_X.append(X[i])
            testing_y.append(y[i])
        res = SearchFireFly(training_X, training_y, testing_X, testing_y)
        print("check done")
        if res["best_objective_function_value"] < best_acc:
            best_acc = res["best_objective_function_value"]
            best_c = res["best_decision_variable_values"][0]
            best_gamma = res["best_decision_variable_values"][1]

        print("=============================")
        fold += 1


    print('BEST PARAMETERS')
    print([best_c, best_gamma])
    # TESTING
    # res2,report=SearchFireFly(X,y,X_test,y_test)
    res2 = SearchingParameters(X, y, X_test, y_test, [best_c, best_gamma])
    REPORT += "TESTING\n"
    REPORT += "Best Parameters : " + str(best_c) + "," + str(best_gamma) + "\n"
    REPORT += "Best MSE : " + str(res2) + "\n"
    print(REPORT)
    model1 = LSSVR(kernel='rbf', gamma=best_gamma, C=best_c)  # gamma=0.003, C=100
    model1.fit(x_train, y_train)
    y_hat1 = model1.predict(x_test)

    plt.plot(y_test)
    plt.plot(y_hat1)
    plt.legend(["test", "predicted"])
    plt.title("Firefly Algorithm + LSSVR test and predicted")
    plt.show()


KFOLDS(15)


# START SEARCHING END==================================================
