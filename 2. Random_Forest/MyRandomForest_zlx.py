# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random
import collections
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
import itertools


class TreeNode(object):
    """定义决策树中的一个结点"""
    def __init__(self):
        self.split_feature = None # 若不为叶子结点，则代表分类特征；否则为None
        self.split_value = None   # 若有分类特征，则代表该特征的二分点位置
        self.left_tree = None     # 若不为叶子结点，则为左子树；否则为None
        self.right_tree = None    # 若不为叶子结点，则为右子树；否则为None
        self.leaf_label = None    # 若为叶节点，则代表该叶节点的分类；否则为None

    def calc_predict_value(self, features):
        """递归寻找样本所属的叶子节点（即预测分类）"""
        # 为叶子结点
        if self.leaf_label is not None:
            return self.leaf_label
        # 不为叶子结点，当前样本该特征取值小于二分点，去左子树寻找
        elif features[self.split_feature] <= self.split_value:
            return self.left_tree.calc_predict_value(features)
        # 不为叶子结点，当前样本该特征取值大于二分点，去右子树寻找
        else:
            return self.right_tree.calc_predict_value(features)


class RandomForestClassifier(object):
    def __init__(self, n_estimators=10, max_depth=-1, min_samples_split=2, min_samples_leaf=1,
                 min_split_entropy=0.0, colsample=0.5, rowsample=0.8, random_state=None):
        """
        随机森林参数
        ----------
        n_estimators:      树数量
        max_depth:         树深度，-1表示不限制深度
        min_samples_split: 节点分裂所需的最小样本数量，小于该值节点终止分裂
        min_samples_leaf:  叶子节点最少样本数量，小于该值叶子被合并
        min_split_gain:    分裂所需的最小增益，小于该值节点终止分裂
        colsample:         列采样比例
        subsample:         行采样比例
        random_state:      随机种子，设置之后每次生成的n_estimators个样本集不会变，确保实验可重复
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth if max_depth != -1 else float('inf')
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_split_entropy = min_split_entropy
        self.colsample = colsample
        self.rowsample = rowsample
        self.random_state = random_state
        self.trees = dict()
        self.feature_importances_ = dict()

    def fit(self, features, label):
        """模型训练入口"""
        if self.random_state:
            random.seed(self.random_state)
        # 从range(self.n_estimators)这个list中返回self.n_estimators个元素组成的list
        random_state_stages = random.sample(range(self.n_estimators), self.n_estimators)

        # 按照行列采样比例，设置行列采样数量
        self.colsample = int(self.colsample * len(features.columns))
        self.rowsample = int(self.rowsample * len(features))

        for tree_num in range(self.n_estimators):
            print(("tree_num: " + str(tree_num+1)).center(40, '-'))

            # 随机选择行和列，每棵树均设置不同的seed保证行列选择都是不同的
            random.seed(random_state_stages[tree_num])
            random_row_indexes = random.sample(range(len(features)), int(self.rowsample))   # 行
            random_col_indexes = random.sample(features.columns.tolist(), self.colsample)   # 列
            features_random = features.loc[random_row_indexes, random_col_indexes].reset_index(drop=True)
            label_random = label.loc[random_row_indexes, :].reset_index(drop=True)

            tree_root = self._fit(features_random, label_random, depth=0)
            self.trees[tree_num] = tree_root

    def _fit(self, features, label, depth):
        """递归建立决策树"""
        tree_node = TreeNode()
        # 如果当前节点下的样本不需要分类（类别全都一样/样本小于分裂要求的最小样本数量），
        # 或者树的深度超过最大深度，则选取出现次数最多的类别。终止分裂
        if (len(label['label'].unique()) <= 1 or len(label) <= self.min_samples_split) \
                or (depth >= self.max_depth):
            tree_node.leaf_label = self.calc_leaf_value(label['label'])
            return tree_node

        # 当前结点需要分裂
        left_features, right_features, left_label, right_label, best_split_feature, best_split_value, best_split_entropy = self.best_split(features, label)
        # 如果父节点分裂后，左叶子节点/右叶子节点样本小于设置的叶子节点最小样本数量，则该父节点终止分裂
        if len(left_features) <= self.min_samples_leaf or \
                len(right_features) <= self.min_samples_leaf or \
                best_split_entropy <= self.min_split_entropy:
            tree_node.leaf_label = self.calc_leaf_value(label['label'])
            return tree_node
        # 否则可以分裂
        else:
            # 如果分裂的时候用到该特征，则该特征的importance加1；为属于非叶子结点的属性赋值
            self.feature_importances_[best_split_feature] = self.feature_importances_.get(best_split_feature, 0) + 1
            tree_node.split_feature = best_split_feature
            tree_node.split_value = best_split_value
            tree_node.left_tree = self._fit(left_features, left_label, depth + 1)
            tree_node.right_tree = self._fit(right_features, right_label, depth + 1)
            return tree_node

    def best_split(self, features, label):
        """寻找最好的样本划分方式，找到最优分裂特征、分裂点、分裂交叉熵，并根据其对样本进行划分"""
        best_split_feature = None
        best_split_value = None
        best_split_entropy = 1
        for feature in features.columns:
            # 训练集每个维度的特征个数都不多，可以直接选取它们作为分位点
            split_values_list = sorted(features[feature].unique().tolist())
            # 对可能的分裂阈值求分裂增益，选取增益最大的阈值
            for split_value in split_values_list:
                left_label = label[features[feature] <= split_value]
                right_label = label[features[feature] > split_value]
                left_ratio = 1.0 * len(left_label) / len(label)
                right_ratio = 1.0 * len(right_label) / len(label)
                split_entropy = self.calc_gini(left_label['label'], left_ratio) + \
                                self.calc_gini(right_label['label'], right_ratio)
                if split_entropy < best_split_entropy:
                    best_split_feature = feature
                    best_split_value = split_value
                    best_split_entropy = split_entropy
        # 划分样本
        left_features = features[features[best_split_feature] <= best_split_value]
        left_label = label[features[best_split_feature] <= best_split_value]
        right_features = features[features[best_split_feature] > best_split_value]
        right_label = label[features[best_split_feature] > best_split_value]
        return left_features, right_features, left_label, right_label, \
               best_split_feature, best_split_value, best_split_entropy

    @staticmethod
    def calc_leaf_value(label):
        """选择样本中出现次数最多的类别作为叶子节点取值"""
        label_counts = collections.Counter(label)
        most_key = 1
        most_count = 0
        for key in label_counts.keys():
            if label_counts[key] > most_count:
                most_key = key
                most_count = label_counts.get(key)
        return most_key

    @staticmethod
    def calc_gini(label, ratio):
        """采用基尼指数作为交叉熵的近似估计，计算每棵子树的基尼指数"""
        gini = 1
        label_counts = collections.Counter(label)
        for key in label_counts:
            prob = label_counts[key] * 1.0 / len(label)
            gini -= prob ** 2
        return gini * ratio

    def predict(self, features):
        """输入样本，预测所属类别"""
        pred_result = []
        for _, row in features.iterrows():
            trees_pred = []
            # 每棵树独立预测，投票决定预测的标签
            for _, tree in self.trees.items():
                trees_pred.append(tree.calc_predict_value(row))
            pred_label_counts = collections.Counter(trees_pred)
            most_key = 1
            most_count = 0
            for key in pred_label_counts.keys():
                if pred_label_counts[key] > most_count:
                    most_key = key
                    most_count = pred_label_counts[key]
            pred_result.append(most_key)
        return np.array(pred_result)


def save_model(model, filename):
    file = open(filename, 'wb')
    pickle.dump(model, file)
    file.close()


def load_model(filename):
    file = open(filename, 'rb')
    return pickle.load(file)


if __name__ == '__main__':
    train = False
    if train:
        x = pd.read_csv("x_train.csv").drop('index', axis=1).reset_index(drop=True)
        y = pd.read_csv("y_train.csv").drop('index', axis=1).reset_index(drop=True)
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)

        x_train = x_train.reset_index(drop=True)
        x_val = x_val.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

        parameter = []

        n_estimators_all = [15]
        max_depth_all = [-1]
        min_samples_split_all = [6]
        min_samples_leaf_all = [2]
        min_split_entropy_all = [0.0]
        colsample_all = [0.5]
        rowsample_all = [0.8]
        '''
        n_estimators_all = [5, 10, 15, 20, 30]
        max_depth_all = [-1, 10, 20]
        min_samples_split_all = [6]
        min_samples_leaf_all = [2]
        min_split_entropy_all = [0.0, 0.05, 0.1]
        colsample_all = [0.2, 0.35, 0.5, 0.65, 0.8]
        # rowsample_all = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        rowsample_all = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        '''
        for (n_estimators, max_depth, min_samples_split, min_samples_leaf, min_split_entropy, colsample, rowsample) in \
                itertools.product(n_estimators_all, max_depth_all, min_samples_split_all, min_samples_leaf_all, min_split_entropy_all, colsample_all, rowsample_all):
            print('------parameters: ', n_estimators, max_depth, min_samples_split, min_samples_leaf, min_split_entropy, colsample, rowsample)
            RfClf = RandomForestClassifier(n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf,
                                           min_split_entropy=min_split_entropy,
                                           colsample=colsample,
                                           rowsample=rowsample,
                                           random_state=413)
            RfClf.fit(x_train, y_train)
            save_model(RfClf, 'MyRfModel.txt')
            train_acc = metrics.accuracy_score(y_train, RfClf.predict(x_train))
            val_acc = metrics.accuracy_score(y_val, RfClf.predict(x_val))
            train_f1 = metrics.f1_score(y_train, RfClf.predict(x_train))
            val_f1 = metrics.f1_score(y_val, RfClf.predict(x_val))
            print('train acc: ', train_acc)
            print('val acc: ', val_acc)
            print('train f1: ', train_f1)
            print('val f1: ', val_f1)
            if val_acc > 0.97 and val_f1 > 0.97:
                parameter.append([n_estimators, max_depth, min_samples_split, min_samples_leaf, min_split_entropy, colsample, rowsample,
                                  train_acc, val_acc, train_f1, val_f1])
        pd.DataFrame(np.array(parameter), columns=['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'min_split_entropy', 'colsample_bytree', 'rowsample', 'train_acc', 'val_acc', 'train_f1', 'val_f1']).\
            to_csv('parameter.csv')

    else:
        '''
        RfClf = load_model('MyRfModel.txt')
        x = pd.read_csv("x_train.csv").drop('index', axis=1).reset_index(drop=True)
        y = pd.read_csv("y_train.csv").drop('index', axis=1).reset_index(drop=True)
        array = RfClf.predict(x)
        print('test acc: ', metrics.accuracy_score(y, array))
        print('test f1: ', metrics.f1_score(y, array))
        df = pd.DataFrame(array, columns=['label'])
        df.to_csv('prediction.csv', index_label=None, index=False)
        '''
        RfClf = load_model('MyRfModel.txt')
        x = pd.read_csv("x_test.csv").reset_index(drop=True)
        array = RfClf.predict(x.drop('index', axis=1))
        x['label'] = array
        df = x[['index', 'label']]
        df.to_csv('17373157-rf.csv', index_label=None, index=False)

        '''    
        y = pd.read_csv("y_train.csv").drop('index', axis=1).reset_index(drop=True)
        print('test acc: ', metrics.accuracy_score(y, array))
        print('test f1: ', metrics.f1_score(y, array))
        '''

        RfClf = load_model('MyRfModel.txt')
        x = pd.read_csv("x_train.csv").reset_index(drop=True)
        array = RfClf.predict(x.drop('index', axis=1))
        x['label'] = array
        df = x[['index', 'label']]
        df.to_csv('y_train_val.csv', index_label=None, index=False)

        y = pd.read_csv("y_train.csv").drop('index', axis=1).reset_index(drop=True)
        print('test acc: ', metrics.accuracy_score(y, array))
        print('test f1: ', metrics.f1_score(y, array))


