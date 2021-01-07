import pickle
import numpy as np


class NN(object):
    def __init__(self,
                 hidden_dims=(512, 256),
                 datapath='cifar10.pkl',
                 n_classes=10,
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=1000,
                 seed=None,
                 activation="relu",
                 init_method="glorot",
                 normalization=False
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if datapath is not None:
            u = pickle._Unpickler(open(datapath, 'rb'))
            u.encoding = 'latin1'
            self.train, self.valid, self.test = u.load()
            if normalization:
                self.normalize()
        else:
            self.train, self.valid, self.test = None, None, None

    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.weights = {}
        # self.weights is a dictionnary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            # WRITE CODE HERE
            upper_limit = np.sqrt(6/(all_dims[layer_n-1] + all_dims[layer_n]))
            lower_limit = -upper_limit
            self.weights[f"W{layer_n}"] = np.random.uniform(low=lower_limit, high=upper_limit, size = (all_dims[layer_n-1], all_dims[layer_n]))
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

    def relu(self, x, grad=False):
        if grad:
            value = np.where(x > 0, 1, 0)
        else:
            value = np.maximum(x , 0)
        return value

    def sigmoid(self, x, grad=False):
        if grad:
            value = np.exp(-x) * (1 + np.exp(-x))**-2
        else :
            value = 1 / (1 + np.exp(-x))
        return value

    def tanh(self, x, grad=False):
        if grad:
            value = 4 / ((np.exp(-x) + np.exp(x)) ** 2)
        else:
            value = (np.exp(x) - np.exp(-x)) / (np.exp(-x) + np.exp(x))
        return value

    def leakyrelu(self, x, grad=False):
        alpha = 0.01
        if grad:
            value = np.where(x > 0, 1, alpha)
        else:
            value = np.where(x > 0, x, alpha * x)
        return value

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            return self.relu(x, grad)
        elif self.activation_str == "sigmoid":
            return self.sigmoid(x, grad)
        elif self.activation_str == "tanh":
            return self.tanh(x, grad)
        elif self.activation_str == "leakyrelu":
            return self.leakyrelu(x, grad)
        else:
            raise Exception("invalid")

    def softmax(self, x):
        # Remember that softmax(x-C) = softmax(x) when C is a constant.
        # WRITE CODE HERE
        shape = x.shape
        value = []
        if len(shape) == 1:
            C = np.amax(x)
            x = x - C
            denom = np.sum(np.exp(x))
            value = np.exp(x) / denom

        else:
            for vect in x:
                C = np.amax(vect)
                vect = vect - C
                denom = np.sum(np.exp(vect))
                value_row = np.exp(vect) / denom
                value.append(value_row) 
        
        return value

    def forward(self, x):
        cache = {"Z0": x}
        for n_layer in range(1, self.n_hidden + 2):
            cache[f"A{n_layer}"] = np.matmul(cache[f"Z{n_layer-1}"], self.weights[f"W{n_layer}"]) + self.weights[f"b{n_layer}"]
            if(n_layer == self.n_hidden+1):
                cache[f"Z{n_layer}"] = self.softmax(cache[f"A{n_layer}"])
            else:
                cache[f"Z{n_layer}"] = self.activation(cache[f"A{n_layer}"])
        # cache is a dictionnary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
        # WRITE CODE HERE
        return cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}

        for n_layer in range(self.n_hidden + 1, 0, -1):
            if(n_layer == self.n_hidden + 1):
                grads[f"dA{n_layer}"] = output - labels
                grads[f"dW{n_layer}"] = (cache[f"Z{n_layer-1}"].transpose() @ grads[f"dA{n_layer}"])/len(labels)
            
            else:
                grads[f"dZ{n_layer}"] = grads[f"dA{n_layer + 1}"] @ self.weights[f"W{n_layer + 1}"].transpose()
                grads[f"dA{n_layer}"] = grads[f"dZ{n_layer}"] * self.activation(cache[f"A{n_layer}"], True)
                grads[f"dW{n_layer}"] = (cache[f"Z{n_layer - 1}"].transpose() @ grads[f"dA{n_layer}"]) / len(grads[f"dA{n_layer}"])
                
            grads[f"db{n_layer}"] = np.mean(grads[f"dA{n_layer}"], axis=0, keepdims=True)
            

        # grads is a dictionnary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
        # WRITE CODE HERE
        return grads

    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            self.weights[f"W{layer}"] = self.weights[f"W{layer}"] - self.lr*grads[f"dW{layer}"]
            self.weights[f"b{layer}"] = self.weights[f"b{layer}"] - (self.lr*grads[f"db{layer}"])

    def one_hot(self, y):
        matrix = np.zeros((len(y), self.n_classes))
        for i in range(len(y)):
            matrix[i][y[i]] = 1
        return matrix

    def loss(self, prediction, labels):
        prediction = np.array(prediction)
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon

        return -np.sum(np.log(prediction) * labels, axis=1).mean()

    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = self.one_hot(y_train)
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                caches = self.forward(minibatchX)
                grads = self.backward(caches, minibatchY)
                self.update(grads)
            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
        return test_loss, test_accuracy

    def normalize(self):
        # WRITE CODE HERE
        # compute mean and std along the first axis
        pass

    
