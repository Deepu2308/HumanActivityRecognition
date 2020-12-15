# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 06:29:06 2020

@author: deepu
"""

from sklearn import tree


models = {
    'decision_tree_gini'    : tree.DecisionTreeClassifier(),
    'decision_tree_entropy' : tree.DecisionTreeClassifier(criterion='entropy')
    }