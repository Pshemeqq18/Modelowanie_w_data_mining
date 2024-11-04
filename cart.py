import pandas as pd
import numpy as np
import math

class cart_tree():

    def __init__(self, pred, target):
        self.pred = pred
        self.target = target

    def train_tree(self, data, max_depth = 5):
        self.data = data
        self.max_depth = max_depth
        self.tree = self.__calc_node(data, 0, {x:[] for x in self.pred})
        return self.tree

    def __calc_node(self, subdataset, depth, used_values):
        if len(subdataset.index) <= 1:
            return subdataset[self.target].mean()
        if depth >= self.max_depth:
            return subdataset[self.target].mean()
        condition, used_values = self.__calc_condition(subdataset, used_values)
        if condition == {}:
            return subdataset[self.target].mean()
        node = {
            "var": condition["var"],
            "val": condition["val"],
            "left": self.__calc_node(subdataset[subdataset[condition["var"]] < condition["val"]], depth + 1, used_values),
            "right": self.__calc_node(subdataset[subdataset[condition["var"]] >= condition["val"]], depth + 1, used_values)
        }
        return node

    def __calc_condition(self, subset, used_values):
        optimal_condition = {}
        for x in self.pred:
            if x not in subset.columns:
                raise Exception("Predictior: " + str(x)+ " not found in the provided dataset")
            unique_subset = [arg for arg in subset[x].unique() if arg not in used_values[x]]
            for y in unique_subset:
                left_var = subset[subset[x] < y][self.target].var()
                right_var = subset[subset[x] >= y][self.target].var()
                temp_var = left_var+right_var
                if optimal_condition == {}:
                    optimal_condition = {
                        "var": x,
                        "val": y,
                        "variance": temp_var
                    }
                elif optimal_condition["variance"] > temp_var:
                    optimal_condition = {
                        "var": x,
                        "val": y,
                        "variance": temp_var
                    }
        if optimal_condition == {}:
            return {}, used_values
        used_values[optimal_condition["var"]].append(optimal_condition["val"])
        return optimal_condition, used_values

    def test_tree(self, test_data):
        if type(self.tree) == dict:
            test_data[("estimated_"+str(self.target))] = test_data.apply(lambda x: self.__go_through_tree(x, self.tree, self.data), axis=1)
            return test_data
        else:
            return self.tree

    def __go_through_tree(self, values, node, target_data):
        if values[node["var"]] >= node["val"]:
            if type(node["right"]) is not dict:
                if np.isnan(node["right"]) == True:
                    if type(node["left"]) is not dict:
                        return node["left"]
                    else:
                        return self.__go_through_tree(values, node["left"], target_data[target_data[node["var"]] < node["val"]])
                return node["right"]
            else:
                return self.__go_through_tree(values, node["right"], target_data[target_data[node["var"]] >= node["val"]])
        else:
            if type(node["left"]) is not dict:
                if np.isnan(node["left"]) == True:
                    if type(node["right"]) is not dict:
                        return node["right"]
                    else:
                        return self.__go_through_tree(values, node["right"], target_data[target_data[node["var"]] < node["val"]])
                return node["left"]
            else:
                return self.__go_through_tree(values, node["left"], target_data[target_data[node["var"]] < node["val"]])

