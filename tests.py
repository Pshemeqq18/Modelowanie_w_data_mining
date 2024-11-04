import cart
import pytest
import pandas as pd

def single_row_test():
    df_train = pd.DataFrame({'arg_1': [1], 'arg_2': [2], 'label_1': [5]})
    df_test = pd.DataFrame({'arg_1': [3, 4], 'arg_2': [7, 8], 'label_1': [2, 3]})
    tree = cart.cart_tree(['arg_1', 'arg_2'], 'label_1')
    tree.train_tree(df_train)
    test = tree.test_tree(df_test)
    assert test == 5

def two_groups_test():
    df_train = pd.DataFrame({'arg_1': [3, 2, 1, 12, 11, 10], 'arg_2': [1, 2, 3, 10, 11, 12], 'label_1': [1, 1, 1, 10, 10, 10]})
    tree = cart.cart_tree(['arg_1', 'arg_2'], 'label_1')
    tree.train_tree(df_train, 1)
    assert tree.tree['left'] == 1 and tree.tree['right'] == 10

def default_depth_test():
    df = pd.DataFrame({'arg_1': [1, 2, 3, 4, 5, 6], 'arg_2': [1, 2, 3, 4, 5, 6], 'label_1': [1, 2, 3, 4, 5, 6]})
    tree = cart.cart_tree(['arg_1', 'arg_2'], 'label_1')
    tree.train_tree(df)
    test = tree.test_tree(df)
    assert test['estimated_label_1'].max() == 5.5

def match_values_test():
    df = pd.DataFrame({'arg_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'arg_2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'label_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    tree = cart.cart_tree(['arg_1', 'arg_2'], 'label_1')
    tree.train_tree(df, 2000)
    test = tree.test_tree(df)
    assert test['estimated_label_1'].to_list() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def multidimensional_test():
    df_train = pd.DataFrame({'arg_1': [1, 2, 3], 'arg_2': [1, 2, 3], 'arg_3': [1, 2, 3], 'arg_4': [1, 2, 3], 'label_1': [1, 2, 3]})
    df_test = pd.DataFrame({'arg_1': [1], 'arg_2': [1], 'arg_3': [1], 'arg_4': [1], 'label_1': [1]})
    tree = cart.cart_tree(['arg_1', 'arg_2', 'arg_3', 'arg_4'], 'label_1')
    tree.train_tree(df_train)
    test = tree.test_tree(df_test)
    assert test['estimated_label_1'].all() == 1

single_row_test()
two_groups_test()
default_depth_test()
match_values_test()
multidimensional_test()