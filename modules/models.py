import numpy as np

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

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

class GradientBoost:
    def __init__(self, learning_rate, n_estimators):
        self.nu = learning_rate
        self.T = n_estimators

        self.best_metric_split = 0
        self._F0_x = 0

        self._trees_list = list()
        self._gammas_list = list()
        self.metric_list = list()

    def _convert_indices_to_residuals(self, leaves_indices: np.ndarray, residuals_dict: np.ndarray):
        residuals_array_len = np.max(list(residuals_dict.keys())) + 1
        residuals_array = np.zeros(residuals_array_len)
        residuals_array[list(residuals_dict.keys())] = list(residuals_dict.values())

        return np.take(a=residuals_array, indices=leaves_indices)
    
    def _predict(self, X, split="all"):
        assert split in ["all", "best"]

        if split == "best":
            assert len(self._trees_list) > 0, "Empty trees list"

            trees_list = self._trees_list[:self.best_metric_split]
            gammas_list = self._gammas_list[:self.best_metric_split]
        else:
            trees_list = self._trees_list
            gammas_list = self._gammas_list

        Ft_x = np.full(len(X), self._F0_x)

        for t, tree_t in enumerate(trees_list):
            leaves_indices = tree_t.apply(X)
            res_t = self._convert_indices_to_residuals(leaves_indices, gammas_list[t]) # get residuals correspond to leaves indices of the samples

            Ft_x += self.nu * res_t

        return Ft_x
    
    def validate(self):
        pass

class GradientBoostRegressor(GradientBoost):
    def __init__(self, learning_rate=0.1, n_estimators=10, **tree_kwargs):
        super().__init__(learning_rate=learning_rate, n_estimators=n_estimators)
        self.tree_kwargs = tree_kwargs

    def __get_gamma_for_all_leaves(self, leaves_indices: np.ndarray, r_it: np.ndarray):
        ret = dict()

        for j in np.sort(np.unique(leaves_indices)):
            mask = leaves_indices == j # get samples indices at the same leaf node
            gamma_j = np.mean(r_it[mask])

            ret[j] = gamma_j

        return ret

    def predict(self, X, split="all"):
        return self._predict(X, split)
    
    def validate(self, val_X, val_Y, split="all"):
        preds = self.predict(val_X, split=split)

        return mean_squared_error(val_Y, preds)

    def fit(self, train_X, train_Y, val_X, val_Y):
        # empty previous run
        if len(self._trees_list) > 0:
            self._trees_list = list()
            self._gammas_list = list()
            self.metric_list = list()
        
        self._F0_x = np.round(np.mean(train_Y), decimals=2) # remember the first avg value
        Ft_x = np.full(len(train_X), self._F0_x)

        for t in range(self.T):
            r_it = train_Y - Ft_x

            tree_t = DecisionTreeRegressor(criterion='friedman_mse', **self.tree_kwargs)
            tree_t.fit(train_X, r_it)

            leaves_indices = tree_t.apply(train_X) # get leaf region indices for all samples
            gamma_t = self.__get_gamma_for_all_leaves(leaves_indices, r_it)
            Ft_x += self.nu * self._convert_indices_to_residuals(leaves_indices, gamma_t) # update F(x) with this run residuals

            self._trees_list.append(tree_t)
            self._gammas_list.append(gamma_t)

            # calculate metric for this iteration
            self.metric_list.append(self.validate(val_X, val_Y))

        # update index of trees split with best metric
        self.best_metric_split = np.argmin(self.metric_list) + 1

class GradientBoostBinaryClassifier(GradientBoost):
    def __init__(self, learning_rate=0.1, n_estimators=100, **tree_kwargs):
        super().__init__(learning_rate=learning_rate, n_estimators=n_estimators)
        self.tree_kwargs = tree_kwargs

    def __probs_2_logodds(self, probs: np.ndarray):
        assert np.min(probs)>=0 and np.max(probs)<=1, "Invalid probs range"

        return np.round(np.log(probs/(1-probs + 1e-6)), decimals=2)

    def __logodds_2_probs(self, log_odds: np.ndarray):
        return np.round(np.exp(log_odds) / (1+np.exp(log_odds)), decimals=2)

    def __get_gamma_for_all_leaves(self, leaves_indices: np.ndarray, r_it: np.ndarray, Ft_x: np.ndarray):
        assert len(leaves_indices.shape)==1 and len(leaves_indices)==len(Ft_x) 

        ret = dict()

        for j in np.unique(leaves_indices):
            mask = leaves_indices == j # get samples indices at the same leaf node

            converted_Ft = self.__logodds_2_probs(Ft_x)
            gamma_j = np.sum(r_it[mask]) / np.sum(converted_Ft[mask]*(1-converted_Ft[mask]))
            
            ret[j] = gamma_j

        return ret

    def predict(self, X, split="all"):
        pred_logodds = self._predict(X, split)
        
        return (self.__logodds_2_probs(pred_logodds) > 0.5) * 1 # convert probs to {0,1} classes
    
    def validate(self, val_X, val_Y, split="all"):
        preds = self.predict(val_X, split=split)

        return accuracy_score(val_Y, preds)

    def fit(self, train_X, train_Y, val_X, val_Y):
        # empty previous run
        if len(self._trees_list) > 0:
            self._trees_list = list()
            self._gammas_list = list()
            self.metric_list = list()

        self._F0_x = self.__probs_2_logodds(np.sum(train_Y)/len(train_Y)) # remember the first avg value
        Ft_x = np.full(len(train_X), self._F0_x)

        for t in range(self.T):
            r_it = np.round(train_Y - self.__logodds_2_probs(Ft_x), decimals=2)

            if np.sum(np.abs(r_it)<0.05) > np.ceil(0.8*len(train_X)):
                print(f"Early stopped at {t}-th tree")
                break

            tree_t = DecisionTreeRegressor(criterion='friedman_mse', **self.tree_kwargs)
            tree_t.fit(train_X, r_it)

            leaves_indices = tree_t.apply(train_X) # get leaf region indices for all samples
            gamma_t = self.__get_gamma_for_all_leaves(leaves_indices, r_it, Ft_x)
            Ft_x += self.nu * self._convert_indices_to_residuals(leaves_indices, gamma_t) # update F(x) with this run residuals

            self._trees_list.append(tree_t)
            self._gammas_list.append(gamma_t)

            # calculate metric for this iteration
            self.metric_list.append(self.validate(val_X, val_Y))

        # update index of trees split with best metric
        self.best_metric_split = np.argmax(self.metric_list) + 1