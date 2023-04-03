import io
from collections import Counter
from pip._internal import main as pipmain


import numpy as np
import pandas as pd
import scipy.io
import sklearn.model_selection
import sklearn.tree
from numpy import genfromtxt
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import mode


# !pip install pydot

import pydot

eps = 1e-5  # a small number


class DecisionTree:
    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes
        
    def accuracy(self, X, y):
        yhat = self.predict(X)
        return np.sum(yhat == y) / float(len(y))

    @staticmethod
    def information_gain(X, y, thresh):
        
        # Split the samples using the threshold value
        idx_left = X < thresh
        idx_right = X >= thresh
        y_left = y[idx_left]
        y_right = y[idx_right]
        
        # Calculate the Gini impurity of each split
        impurity_left = DecisionTree.gini_impurity(X, y_left, thresh)
        impurity_right = DecisionTree.gini_impurity(X, y_right, thresh)
        
        # Calculate the weighted average of the Gini impurities
        n_total = len(y)
        n_left = len(y_left)
        n_right = len(y_right)
        impurity_avg = (n_left / n_total) * impurity_left + (n_right / n_total) * impurity_right
        
        # Calculate the Gini impurity of the parent node
        impurity_parent = DecisionTree.gini_impurity(X, y, thresh)
        
        # Calculate the information gain
        information_gain = impurity_parent - impurity_avg
        
        return information_gain


    @staticmethod
    def gini_impurity(X, y, thresh):
        
        
        H = 0
        for i in np.unique(y):
            p = np.where(y == i, 1, 0)
            p_c = np.sum(p) / len(y)
            H -= p_c * np.log2(p_c)
        return H




    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        if self.max_depth > 0:
            # compute entropy gain for all single-dimension splits,
            # thresholding with a linear interpolation of 10 values
            gains = []
            # The following logic prevents thresholding on exactly the minimum
            # or maximum values, which may not lead to any meaningful node
            # splits.
            thresh = np.array([
                np.linspace(np.min(X[:, i]) + eps, np.max(X[:, i]) - eps, num=10)
                for i in range(X.shape[1])
            ])
            for i in range(X.shape[1]):
                gains.append([self.information_gain(X[:, i], y, t) for t in thresh[i, :]])

            gains = np.nan_to_num(np.array(gains))
            self.split_idx, thresh_idx = np.unravel_index(np.argmax(gains), gains.shape)
            self.thresh = thresh[self.split_idx, thresh_idx]
            X0, y0, X1, y1 = self.split(X, y, idx=self.split_idx, thresh=self.thresh)
            if X0.size > 0 and X1.size > 0:
                self.left = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.left.fit(X0, y0)
                self.right = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.right.fit(X1, y1)
            else:
                self.max_depth = 0
                self.data, self.labels = X, y
                self.pred = stats.mode(y).mode[0]
        else:
            self.data, self.labels = X, y
            self.pred = stats.mode(y).mode[0]
        return self

    def predict(self, X):
        if self.max_depth == 0:
            return self.pred * np.ones(X.shape[0])
        else:
            X0, idx0, X1, idx1 = self.split_test(X, idx=self.split_idx, thresh=self.thresh)
            yhat = np.zeros(X.shape[0])
            yhat[idx0] = self.left.predict(X0)
            yhat[idx1] = self.right.predict(X1)
            return yhat

    def __repr__(self):
        if self.max_depth == 0:
            return "%s (%s)" % (self.pred, self.labels.size)
        else:
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                           self.thresh, self.left.__repr__(),
                                           self.right.__repr__())


class BaggedTrees(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            sklearn.tree.DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)
    def accuracy(self, X, y):
        yhat = self.predict(X)
        return np.sum(yhat == y) / float(len(y))
    
    def fit(self, X, y):
        
        for tree in self.decision_trees:
            bagged_samples = np.random.choice(list(range(len(X))), size=len(X), replace=True)
            X_train = X[bagged_samples]
            y_train = y[bagged_samples]
            tree.fit(X_train, y_train)

       
    def predict(self, X):
        
        predictions = []
        for tree in self.decision_trees:
            predictions.append(tree.predict(X))
        return mode(predictions)[0]


class RandomForest(BaggedTrees):
    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        self.n = n  
        self.sample_size = m 
        self.decision_trees = [
            DecisionTree(list(params.values())[0], list(params.values())[1])
            for i in range(self.n)
        ]
    def accuracy(self, X, y):
        yhat = self.predict(X)
        return np.sum(yhat == y) / float(len(y))


class BoostedRandomForest(RandomForest):
    def fit(self, X, y):
        self.w = np.ones(X.shape[0]) / X.shape[0]  # Weights on data
        self.a = np.zeros(self.n)  # Weights on decision trees
        # TODO: implement function
        return self

    def predict(self, X):
        # TODO: implement function
        pass


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # fill_mode = False
    data[data == ''] = '-1'
    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == '-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack([np.array(data, dtype=float), np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for i in range(data.shape[-1]):
            mode = stats.mode(data[((data[:, i] < -1 - eps) +
                                    (data[:, i] > -1 + eps))][:, i]).mode[0]
            data[(data[:, i] > -1 - eps) * (data[:, i] < -1 + eps)][:, i] = mode

    return data, onehot_features


def evaluate(clf):
    print("Cross validation", sklearn.model_selection.cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [(features[term[0]], term[1]) for term in counter.most_common()]
        print("First splits", first_splits)


# +
if __name__ == "__main__":
    dataset = "titanic"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    
    N = 100
    if dataset == "titanic":
        # Load titanic data
        path_train = './dataset/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None, encoding=None)
        path_test = './dataset/titanic/titanic_test_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None, encoding=None)
        y = data[1:, -1]  # label = survived
        class_names = ["Died", "Survived"]
        labeled_idx = np.where(y != '')[0]

        y = np.array(y[labeled_idx])
        y = y.astype(float).astype(int)

        #cleaning data
        
       
        
        
        print("\n\nPart (b): preprocessing the titanic dataset")
     
        X, onehot_features = preprocess(data[1:, :-1], onehot_cols=[1,5,7,8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1,5,7,8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, :-1]) + onehot_features
        
        

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription", "creative",
            "height", "featured", "differ", "width", "other", "energy", "business", "message",
            "volumes", "revision", "path", "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis", "square_bracket",
            "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = './dataset/spam/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)
        



# +

print("Features:", features)
print("Train/test size:", X.shape, Z.shape)

print("\n\nPart 0: constant classifier")
print("Accuracy", 1 - np.sum(y) / y.size)

# Basic decision tree
print("\n\nPart (a-b): simplified decision tree")
dt = DecisionTree(max_depth=3, feature_labels=features)
dt.fit(X, y)
print("Predictions", dt.predict(Z)[:100])

print("\n\nPart (c): sklearn's decision tree")
clf = sklearn.tree.DecisionTreeClassifier(random_state=0, **params)
clf.fit(X, y)
evaluate(clf)
out = io.StringIO()

# You may want to install "gprof2dot"
sklearn.tree.export_graphviz(
    clf, out_file=out, feature_names=features, class_names=class_names)
graph = pydot.graph_from_dot_data(out.getvalue())
pydot.graph_from_dot_data(out.getvalue())[0].write_pdf("%s-tree.pdf" % dataset)

# TODO: implement and evaluate!

# +
#4.1

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

dt = DecisionTree(max_depth=3, feature_labels = features)
dt.fit(X, y)

y_test = dt.predict(Z)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)
train_acc = dt.accuracy(X_train, y_train)
test_acc = dt.accuracy(X_test, y_test)
print("DT train: ",train_acc, "test:", test_acc)

#4.2

r_dt = RandomForest(params={'max_depth': 5, 'max_features': 5}, n=100, m=5)
r_dt.fit(X, y)
r_X_train, r_X_test, r_y_train, r_y_test = train_test_split(X, y, test_size=0.40)
r_train_acc = r_dt.accuracy(r_X_train, r_y_train)
r_test_acc = r_dt.accuracy(r_X_test, r_y_test)
y_test = r_dt.predict(Z)
print("RTitanic DT train: ",r_train_acc, "test:", r_test_acc)
# +
#4.5 part 3

test_accs = []
train_accs = []

for i in range(40):
    dt = DecisionTree(max_depth=i,feature_labels=features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    
    
    dt.fit(X, y)
    train_preds = dt.predict(X_train)
    train_accs.append(np.mean(train_preds==y_train))
    
    test_preds = dt.predict(X_test)
    test_accs.append(np.mean(test_preds==y_test))
    
    if i % 10 == 0:
        print("finished predicting with depth",i)

# +
import matplotlib.pyplot as plt

plt.plot(range(40), train_accs, label='Train Accuracy')
plt.plot(range(40), test_accs, label='Test Accuracy')
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy vs. Depth')
plt.legend()
plt.show()

# -

def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1 # Ensures that the index starts at 1
    df.to_csv('submission_s.csv', index_label='Id')


y_test = y_test.reshape(-1,)
results_to_csv(y_test)



