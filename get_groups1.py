from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier
from copy import deepcopy
import numpy as np
from datetime import datetime as dd
import sys

def evaluate_classifier(cl, feat, target):
    scores = []
    max_label = max(target)
    feat = np.array(feat)
    target = np.array(target)
    for train_idx, test_idx in KFold(len(feat), n_folds=10, shuffle=True):
        X_train, Y_train = feat[train_idx], target[train_idx]
        X_test, Y_test = feat[test_idx], target[test_idx]

        solution = deepcopy(cl)
        solution.fit(X_train, Y_train)
        
        current_mean = 0
        for i in xrange(max_label + 1):
            idx = np.array([j for j in xrange(len(Y_test)) if Y_test[j] == i])

            bin_y = [1 if t else 0 for t in Y_test[idx]]
            bin_x = X_test[idx]

            answer = solution.predict(bin_x)

            score = accuracy_score(bin_y, answer)
            print('Absolute score for subgroup %s: %f' % (groups[i][0], score))
            score *= float(len(bin_x)) / len(X_test)

            current_mean += score
            global groups
            print('Score for subgroup %s: %f (number of tasks in this group is %d)' % (groups[i][0], score, len(bin_x)))

        print('Mean score on one fold is %f' % (current_mean))
        scores.append(current_mean)

    print('Overall score is %f' % (sum(scores) / len(scores)))

def get_t(s):
    return dd.strptime(s.strip('"'), "%Y-%m-%d %H:%M")

inp = open('statistics.csv', 'r').read().strip().split('\n')

conv = [0 for i in range(600 * 60 * 60)]

groups = [('1min', (0, 60)), ('1hour', (60, 60 * 60)), ('2hours', (60 * 60, 2 * 60 * 60)),\
        ('6hours', (2 * 60 * 60, 6 * 60 * 60)), ('1day', (6 * 60 * 60, 24 * 60 * 60)),\
        ('3days', (24 * 60 * 60, 3 * 24 * 60 * 60)), ('3days', (3 * 24 * 60 * 60, 600 * 60 * 60))]

cnt = 0
for g in groups:
    for i in xrange(g[1][0], g[1][1]):
        conv[i] = cnt
    cnt += 1

features = []
target = []

for i, x in enumerate(inp[1:]):
    row = x.split('\t')

    a = get_t(row[2])
    b = get_t(row[3])

    tm = int((b - a).total_seconds())

    features.append((int(row[7]), int(row[8])))
    target.append(conv[tm])

desired_time = int(sys.argv[1])
desired_proc = int(sys.argv[2])

classifier = LogisticRegression()  
evaluate_classifier(deepcopy(classifier), features, target)
classifier.fit(features, target)

answer = classifier.predict([(desired_time, desired_proc)])[0]

if answer != 0 and answer != len(groups) - 1:
    print('Your task is going to be in queue >%s and <%s' % (groups[answer - 1][0], groups[answer][0]))
elif answer == len(groups) - 1:
    print('Your task is going to be in queue >%s' % (groups[-1][0]))
else:
    print('Your task is going to be in queue <%s' % groups[answer][0])
