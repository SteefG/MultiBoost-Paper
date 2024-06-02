import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# Group is made for returning the final classifier, it collects the same classifications
# in order to then calculate the weights for that class.
class Group:
    def __init__(self):
        self.index = 0
        self.betas = []

# Creates a MultiBoostClassifier using the psuedocode in Webb (2000)
# Uses sklearn API for simplicity of model evaluation
# T and the base learner are variable
class MultiBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, T = 50, baseLearner = DecisionTreeClassifier()): #From sklearn docs
        self.T = T
        self.baseLearner = baseLearner

    def fit(self, X, y): #Adapted from sklearn documentation
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.size_ = len(y)
        self.betas_, self.classifications_ = self._multiBoost(X, y, self.T)
        return self

    def get_params(self, deep = False): #Adapted from sklearn documentation
        return {"T": self.T}
    
    def set_params(self, **parameters): #Adapted from sklearn documentation
        for parameter, value in parameter.items():
            setattr(self, parameter, value)
        return self

    def _weightReset(self, weightSize):
            weights = np.random.randint(1,1000, size = weightSize)
            weights = -1 * np.log(weights / 1000)
            weights = weights / weights.sum() * weightSize  # Standardise to sum to weight size
            return weights

    # As described in Webb (2000), this creates the premature stopping vector
    def _makeIVector(self, T):
        n = math.floor(math.sqrt(T))
        return [math.ceil(i * T / n) if i < n else T for i in range(1, T+1)]

    # Groups classes for each x and finds class with highest weight
    def _getHeaviestClass(self, classifiedMatrix, betas):
        betaVec = betas
        rowLen = len(classifiedMatrix)
        colLen = len(classifiedMatrix[0])
        output_classes = []

        for i in range(rowLen):
            # Structure groups based on unique index and associated betas with the columns
            groups = []
            for j in range(colLen):
                found_group = True
                for k in range(len(groups)):
                    if classifiedMatrix[i][j] == groups[k].index:
                        groups[k].betas.append(betaVec[j])
                        found_group = False

                if found_group:
                    group = Group()
                    group.index = classifiedMatrix[i][j]
                    group.betas.append(betaVec[j])
                    groups.append(group)

            # Compare weighted sums of each group, based on betas, for each row and return the associated unique index
            total = 0
            final_index = groups[0].index
            for j in range(len(groups)):

                new_total = sum(groups[j].betas)
                if new_total >= total:
                    total = new_total
                    final_index = groups[j].index
            output_classes.append(final_index) 
        return output_classes

    # Implements the MultiBoost training algorithm as proposed by Webb (2000)
    def _multiBoost(self, X, y, T):
        size = len(y)
        weights = np.ones(size)
        k = 1
        betas = []
        classifications = []

        I = self._makeIVector(T)

        for t in range(1, T+1):
            if t == I[k-1]:
                weights = self._weightReset(size)
                k += 1
            
            baseLearner = clone(self.baseLearner) #Cloned to retrain the data
            fitted = baseLearner.fit(X,y, sample_weight = weights)
            classifications.append(fitted)
            result = fitted.predict(X)

            wrongVector = []
            for i in range(0, size):
                if y[i] != result[i]:
                    wrongVector.append(weights[i])

            error = np.sum(wrongVector)/size

            retries = 0
            while error > 0.5:
                if retries > 100: #If the base learner is too weak the program will be stuck here forever
                    print("Base Learner too weak, please try another")
                    exit()
                weights = self._weightReset(size)
                k += 1
                retries += 1
                baseLearner = clone(self.baseLearner) #Cloned to retrain the data
                fitted = baseLearner.fit(X,y, sample_weight = weights)
                classifications.append(fitted)
                result = fitted.predict(X)

                wrongVector = []
                for i in range(0, size):
                    if y[i] != result[i]:
                        wrongVector.append(weights[i])

                error = np.sum(wrongVector)/size
                classifications.pop() #Here we do not use this classification so we remove it

            if error == 0: #If error is small set it to 1e-10 to avoid 0 logarithms
                betas.append(1e-10)
                weights = self._weightReset(size)
                k += 1
                continue
            
            else: # Apply weight boosting
                betas.append(error / (1 - error))
                for i in range(0, size):
                    if y[i] != result[i]:
                        weights[i] /= 2 * error
                    else:
                        weights[i] /= (2 * (1 - error))
                    if weights[i] < 1e-8:
                        weights[i] = 1e-8
        return betas, classifications

    def predict(self, X): #Adapted from sklearn documentation
        check_is_fitted(self)
        X = check_array(X)
        classifiedMatrix = []

        for i in range(self.T):
            classifiedMatrix.append(self.classifications_[i].predict(X))

        classifiedMatrix = np.array(classifiedMatrix).T

        return self._getHeaviestClass(classifiedMatrix, self.betas_)