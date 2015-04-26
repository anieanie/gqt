#!/opt/python-2.7.6/bin/python
# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.cross_validation import KFold
from copy import deepcopy
import numpy as np
import argparse
from datetime import datetime as dd
import sys
import tree

''' Each group is defined by two integers (A, B), both denoting number of seconds. 
    Suppose some task is staying in queue for S seconds. Then the task belongs to group (A, B)
    if A <= S < B. '''
groups = [('1min', (0, 60)), ('1hour', (60, 60 * 60)), ('2hours', (60 * 60, 2 * 60 * 60)),\
        ('6hours', (2 * 60 * 60, 6 * 60 * 60)), ('1day', (6 * 60 * 60, 24 * 60 * 60)),\
        ('3days', (24 * 60 * 60, 3 * 24 * 60 * 60)), ('3days', (3 * 24 * 60 * 60, 600 * 60 * 60))]

def evaluate_classifier(cl, feat, target):
    ''' Prints weighted accuracy score for classifier, averaged over K=10 iterations
        of KFold cross-validation. '''
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
            print('Score for subgroup %s: %f (number of tasks in this group is %d)' % (groups[i][0], score, len(bin_x)))

        print('Mean score on one fold is %f\n' % (current_mean))
        scores.append(current_mean)

    print('Overall score is %f' % (sum(scores) / len(scores)))

def get_t(s):
    ''' Converts time from string representation into seconds.'''
    return dd.strptime(s.strip('"'), "%Y-%m-%d %H:%M")

def load_trainig_set():
    ''' Loads training set from statistics.csv. Parses every task,
        converts it into features tuple. Assigns each task to
        group. '''
    inp = open('statistics.csv', 'r').read().strip().split('\n')

    conv = [0 for i in range(600 * 60 * 60)]

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

    return (target, features)

def main(argv=None):
    if argv == None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
            description=\
                    """
            Данная программа производит машинное обучение на 
            статистике запуска задач в многопроцессорном кластере (statisctics.csv).
            В качестве признаков для обучения берутся две величины: 
                1. Время, запрошенное пользователем;
                2. Количество процессоров, запрошенное пользователем.
            Целевой переменной является время ожидания задачи в очереди.

            На основе полученных данных программа может предсказывать для ставящихся задач примерное
            время их ожидания в очереди.

            В программе предусмотрено два вида классификаторов:
                1. Логистическая регрессия;
                2. Дерево решений.
            Выбор классификатора происходит с помощью аргументов командной строки.

            Также есть возможность оценить методом кросс-валидации качество выполняемой
            классификации (также с помощью аргументов командной строки).

            При использовании дерева решений предусмотрен вывод получившегося дерева,
            которое показывает, как признаки были использованы для классификации.

            Примеры корректного запуска программы:
                python get_groups.py --desired-time 16 --desired-proc 8 --classifier log_reg --evaluate-precision true
                python get_groups.py --desired-time 180 --desired-proc 512 --classifier dtree
                python get_groups.py --desired-time 60 --desired-proc 16 --classifier dtree --print-used-features true
                    """,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--desired-time', dest='time', required=True, help='Время, запрошенное пользователем на запуск задачи')
    parser.add_argument('--desired-proc', dest='proc', required=True, help='Количество процессоров, запрошенное пользователем на запуск задачи')

    parser.add_argument('--classifier', dest='cl_type', required=True,\
            choices=['log_reg', 'dtree'], help='Какой классификатор использовать')
    parser.add_argument('--evaluate-precision', dest='need_crossval', default='false', help='Если указано true, то вывести оценку точности классификатора')
    parser.add_argument('--print-used-features', dest='need_dfs', default='false', help='Если указано true, вывести дерево использованных признаков (только для дерева решений)')

    args = parser.parse_args()

    desired_time = int(args.time)
    desired_proc = int(args.proc)

    target, features = load_trainig_set()

    classifier = None
    if args.cl_type == 'dtree':
        classifier = tree.ClassificationTree(percent_threshold=1, delta_impurity_min=0.001)
    elif args.cl_type == 'log_reg':
        classifier = LogisticRegression()

    if args.need_crossval != 'false':
        evaluate_classifier(deepcopy(classifier), features, target)

    classifier.fit(features, target)
    if args.need_dfs != 'false' and args.cl_type == 'dtree': 
        dfs_ret = classifier.root.dfs()
        dfs_ret = dfs_ret.replace('Nodes', 'Tasks')
        dfs_ret = dfs_ret.replace('feature #0', 'user_timelimit')
        dfs_ret = dfs_ret.replace('feature #1', 'user_processors')
        for i, g in enumerate(groups):
            dfs_ret = dfs_ret.replace('$%d' % i, g[0])
        print dfs_ret


    answer = classifier.predict(np.array([(desired_time, desired_proc)]))[0]

    print '\n\n'
    if answer != 0 and answer != len(groups) - 1:
        print('Your task is going to be in queue >%s and <%s' % (groups[answer - 1][0], groups[answer][0]))
    elif answer == len(groups) - 1:
        print('Your task is going to be in queue >%s' % (groups[-1][0]))
    else:
        print('Your task is going to be in queue <%s' % groups[answer][0])

    return 0

if __name__ == '__main__':
    sys.exit(main())
