import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class DecisionTreeBased_AdaBoost:
    def __init__(self, max_depth, n_estimators):
        self.max_depth = max_depth
        self.T = n_estimators
        self.best_metric_split = 0

        self.trees_list = list()
        self.alphas_list = list()
        self.metric_list = list()
    
    def predict(self, X, split="all"):
        assert split in ["all", "best"]

        if split == "best":
            assert len(self.trees_list) > 0, "Empty trees list"
            
            trees_list = self.trees_list[:self.best_metric_split]
            alphas = np.array(self.alphas_list)[:self.best_metric_split]
        else:
            trees_list = self.trees_list
            alphas = np.array(self.alphas_list)

        outputs = list()

        for tree_t in trees_list:
            outputs.append(tree_t.predict(X))

        outputs = np.array(outputs) # shape (n_trees, n_samples)

        # convert outputs {0,1} to {-1,1} and aggregate to get final preds
        aggregated_outputs = np.sum((outputs*2-1) * np.expand_dims(alphas, axis=1), axis=0) # shape (n_samples,)
        
        return np.clip(np.sign(aggregated_outputs), a_min=0, a_max=1)
    
    def validate(self, y_true, val_X, split="all"):
        assert split in ["all", "best"]

        outputs = self.predict(val_X, split=split)

        return accuracy_score(y_true=y_true, y_pred=outputs)


    def fit(self):
        pass

    def __get_next_iteration(self):
        pass

class AdaBoostRandomized(DecisionTreeBased_AdaBoost):
    def __init__(self, max_depth=1, n_estimators=50):
        super().__init__(max_depth, n_estimators)

    def __get_next_iteration(self, X, Y, outputs):
        def get_alpha(outputs, Y):
            total_error = np.sum(outputs != Y)/len(outputs)

            return 1/2 * np.log((1-total_error)/(total_error+1e-6))

        alpha = get_alpha(outputs, Y)
        samples_num = len(outputs)
        weights = np.full((samples_num), 1/samples_num) # initialize default weight

        mask = np.where(outputs != Y, alpha, -alpha) # assign alpha for incorrect outputs, -alpha for correct outputs
        new_weights = weights * np.exp(mask) # get new weights 
        normalized_weights = new_weights / np.sum(new_weights) # normalize to range [0,1]

        choices = np.random.choice(samples_num, samples_num, p=normalized_weights) # get new dataset 

        return X[choices,:], Y[choices], alpha
    
    def fit(self, X, Y, val_X, val_Y):
        # empty previous run
        if len(self.trees_list) > 0:
            self.trees_list = list()
            self.alphas_list = list()
            self.metric_list = list()

        curr_X = X
        curr_Y = Y

        for t in range(self.T):
            tree_t = DecisionTreeClassifier(max_depth=self.max_depth)
            tree_t.fit(curr_X, curr_Y)

            outputs = tree_t.predict(curr_X)
            curr_X, curr_Y, alpha = self.__get_next_iteration(curr_X, curr_Y, outputs)

            self.trees_list.append(tree_t)
            self.alphas_list.append(alpha)

            # calculate metric for this iteration
            self.metric_list.append(self.validate(y_true=val_Y, val_X=val_X))

        # update index of trees split with best metric
        self.best_metric_split = np.argmax(self.metric_list) + 1

class AdaBoostSampleWeights(DecisionTreeBased_AdaBoost):
    def __init__(self, max_depth=1, n_estimators=50):
        super().__init__(max_depth, n_estimators)

    def __get_next_iteration(self, Y, outputs, sample_weights):
        def get_alpha(outputs, Y, sample_weights):
            total_error = np.sum(sample_weights[outputs != Y]) / np.sum(sample_weights)

            return 1/2 * np.log((1-total_error)/(total_error+1e-6))

        alpha = get_alpha(outputs, Y, sample_weights)

        mask = np.where(outputs != Y, alpha, 1) # assign alpha for incorrect outputs, 1 for correct outputs
        new_weights = sample_weights * np.exp(mask) # get new weights 
        normalized_weights = new_weights / np.sum(new_weights) # normalize to range [0,1]

        return alpha, normalized_weights
    
    def fit(self, X, Y, val_X, val_Y):
        # empty previous run
        if len(self.trees_list) > 0:
            self.trees_list = list()
            self.alphas_list = list()
            self.metric_list = list()

        sample_weights = np.full(len(Y), 1/len(Y)) # init equal weights for all samples

        for t in range(self.T):
            tree_t = DecisionTreeClassifier(max_depth=self.max_depth)
            tree_t.fit(X, Y, sample_weight=sample_weights)

            outputs = tree_t.predict(X)
            alpha, sample_weights = self.__get_next_iteration(Y, outputs, sample_weights)

            self.trees_list.append(tree_t)
            self.alphas_list.append(alpha)

            # calculate metric for this iteration
            self.metric_list.append(self.validate(y_true=val_Y, val_X=val_X))

        # update index of trees split with best metric
        self.best_metric_split = np.argmax(self.metric_list) + 1
