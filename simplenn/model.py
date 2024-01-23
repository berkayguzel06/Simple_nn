from simplenn.engine import Categorical_Cross_Entropy, Activation_Softmax_Loss_CategoricalCrossentropy, Softmax
import pickle
import copy
import numpy as np
import time

class Model:
    def __init__(self):
        self.layers = []
        self.softmax_classifier = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self,loss,optimizer,accuracy):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy

    def forward(self, X, training):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output
    
    def backward(self, output, y):
        if self.softmax_classifier is not None:
            self.softmax_classifier.backward(output, y) 
            self.layers[-1].dinputs = self.softmax_classifier.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)
        self.trainable_layers = []
        for i in range(layer_count):
            if i==0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

            if self.loss is not None:
                self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Softmax) and isinstance(self.loss, Categorical_Cross_Entropy):
            self.softmax_classifier = Activation_Softmax_Loss_CategoricalCrossentropy()

                
    def evaluate(self, X_val, y_val,print_every,batch_size=None):
        validation_steps = 1
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps+=1
        
        self.loss.new_pass()
        self.accuracy.new_pass()
        for steps in range(validation_steps):
            if batch_size is None:
                batch_x = X_val
                batch_y = y_val
            else:
                batch_x = X_val[steps*batch_size:(steps+1)*batch_size]
                batch_y = y_val[steps*batch_size:(steps+1)*batch_size]
            output = self.forward(batch_x, training=False)
            self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        if print_every==0:
                print(f'validation, ' +
                f'acc: {validation_accuracy:.3f}, ' +
                f'loss: {validation_loss:.3f}')
        
    def train(self, X, y, epochs=1, batch_size=None, print_every=1, validation=None):
        ttt = 0 # Total Train Time
        self.accuracy.addPrecision(y)
        train_Steps = 1
        if batch_size is not None:
            train_Steps = len(X) // batch_size
            if train_Steps * batch_size < len(X):
                train_Steps+=1
        
        for epoch in range(1, epochs+1):
            start = time.time()
            self.accuracy.new_pass()
            self.loss.new_pass()
            for steps in range(train_Steps):
                if batch_size is None:
                    batch_x = X
                    batch_y = y
                else:
                    batch_x = X[steps*batch_size:(steps+1)*batch_size]
                    batch_y = y[steps*batch_size:(steps+1)*batch_size]
                output = self.forward(batch_x, training=True)
                data_loss = self.loss.calculate(output, batch_y)
                loss = data_loss
                predictions = self.output_layer_activation.predictions(output)
                acc = self.accuracy.calculate(predictions,batch_y)
                self.backward(output,batch_y)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()
                '''
                if (epoch % print_every)==1 or epoch == epochs - 1:
                    print(
                        f'step: {steps}, ' + f'acc: {acc:.3f}, ' + f'loss: {loss:.3f} '
                        + f'lr: {self.optimizer.currentlr}'
                    )
                '''
            printable = (epoch % print_every)
            if printable==0 or epoch == epochs:
                print("--------------------")
                print(
                    f'epoch: {epoch}, ' + f'acc: {acc:.3f}, ' + f'loss: {loss:.3f} '
                    + f'lr: {self.optimizer.currentlr}'
                )
                printable=0
            if validation is not None:
                self.evaluate(*validation,printable,batch_size=batch_size)
            stop = time.time()
            sub = stop-start
            ttt+=sub
            if printable==0:
                print(f"Time: {sub} s")
                if epoch==epochs:
                    print(f"Total Time: {ttt} s")
        
    def predict(self,X,batch_size=None):
        batch_X = 0
        prediction_steps = 1
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps+=1
        output = []
        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]

            batch_output =self.forward(batch_X, training=False)
            output.append(batch_output)
        return np.vstack(output)
    def get_parameters(self):
        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        return parameters
    def set_parameters(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_parameters(),f)
    def load_parameters(self, path):
        with open(path,"rb") as f:
            self.set_parameters(pickle.load(f))
    def save(self, path):
        model = copy.deepcopy(self)
        model.loss.new_pass()
        model.accuracy.new_pass()

        model.input_layer.__dict__.pop("output",None)
        model.loss.__dict__.pop("dinputs",None)
        for layer in model.layers:
            for prop in ['inputs', 'output', 'dinputs','dweights', 'dbiases']:
                layer.__dict__.pop(prop,None)
        with open(path,"wb") as f:
            pickle.dump(model,f)
    @staticmethod
    def load(path):
        with open(path,"rb") as f:
            model = pickle.load(f)
        return model


class Layer_Input:
    def forward(self, inputs, training):
        self.output = inputs

    
          