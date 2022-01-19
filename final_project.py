import csv
import numpy as np
from sklearn import preprocessing, decomposition, metrics, svm
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


with open('qsar_oral_toxicity.csv', 'rt')as f:
    file = csv.reader(f)
    data = []
    for row in file:
        line = ''.join(row)
        data_line = line.split(';')
        if data_line[-1] == "positive":
            data_line[-1] = 1
        else:
            data_line[-1] = 0
        for i in range(len(data_line)):
            data_line[i] = int(data_line[i])
        data.append(data_line)


data = preprocessing.MinMaxScaler().fit_transform(data)


data_length = len(data)
train_size = int(0.8 * data_length)


data_inputs, data_labels = [], []
for data_line in data:
    data_inputs.append(data_line[:data_length])
    data_labels.append(data_line[-1])


train_inputs = np.array(data_inputs[:train_size])
train_labels = np.array(data_labels[:train_size])
test_inputs = np.array(data_inputs[train_size:])
test_labels = np.array(data_labels[train_size:])


data_inputs = np.concatenate([train_inputs, test_inputs])


data_inputs = decomposition.PCA(n_components=600).fit_transform(data_inputs)


train_inputs = data_inputs[:train_size]
test_inputs = data_inputs[train_size:]


classifying_methods = [
    GaussianNB(),
    KNeighborsClassifier(n_neighbors=3, algorithm='auto'),
    RandomForestClassifier(max_depth=300, criterion='gini'),
    GradientBoostingClassifier(random_state=0, max_depth=1),
    AdaBoostClassifier(n_estimators=100, random_state=0),
    MLPClassifier(random_state=1, max_iter=300),
    svm.SVC(gamma='scale'),
    LogisticRegression(max_iter=1000)

]


def result_computer(train_inputs, train_labels, test_inputs, test_labels, classifier):
    classifier.fit(train_inputs, train_labels)
    prediction_labels = classifier.predict(test_inputs)
    confusion_matrix = metrics.confusion_matrix(test_labels, prediction_labels)
    accuracy_score = metrics.accuracy_score(test_labels, prediction_labels)
    precision_score = metrics.precision_score(test_labels, prediction_labels)
    recall_score = metrics.recall_score(test_labels, prediction_labels)
    f_measure = metrics.f1_score(test_labels, prediction_labels)
    results = {
        'confusion_matrix': confusion_matrix,
        'accuracy_score': accuracy_score,
        'precision_score': precision_score,
        'recall_score': recall_score,
        'f_measure': f_measure
    }
    print("\n\n\n***************{}***************\n\n".format(classifier))
    for ans in results.keys():
            print("---------------{}:--------------\n".format(ans), results[ans])


while True:
    i = int(input("\n Please enter your preferred classifying method or enter 9 to  exit  \n 1:GaussianNB \n 2:KNeighborsClassifier \n 3:RandomForestClassifier \n 4:GradientBoostingClassifier \n 5:AdaBoostClassifier \n 6:MLPClassifier \n 7:SVM \n 8:LogisticRegression \n 9: exit \n"))
    if i == 9:
        exit()
    if not(i >= 0 and i <= 8):
        print("your input in invalid")
        continue

    result_computer(train_inputs, train_labels, test_inputs, test_labels, classifying_methods[i-1])

