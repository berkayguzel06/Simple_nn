import numpy as np

class Loss:
    def remember_trainable_layers(self, trainable_layers):
            self.trainable_layers = trainable_layers

    def calculate(self, output, y):
        losses = self.forward(output, y)
        data_loss = np.mean(losses)
        self.accumulated_sum += np.sum(losses)
        self.accumulated_count += len(losses)
        return data_loss
    
    def calculate_accumulated(self):
        data_loss = self.accumulated_sum / self.accumulated_count
        return data_loss

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


class Categorical_Cross_Entropy(Loss):
    def forward(self, prediction, target):
        samples = len(prediction)
        y_clipped = np.clip(prediction, 1e-7, 1 - 1e-7)
        if len(target.shape) == 1:
            correctConfidences = y_clipped[range(samples), target]
        elif len(target.shape) == 2:
            correctConfidences = np.sum(y_clipped * target, axis=1)
        log = -np.log(correctConfidences)
        return log
    
    def backward(self, dvalues, target):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(target.shape) == 1:
            target = np.eye(labels)[target]
        self.dinputs = -target / dvalues
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy(Loss):
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Binary_Cross_Entropy(Loss):
    def forward(self, predictions, target):
        y_pred_clip = np.clip(predictions, 1e-7, 1 - 1e-7)
        sample_loss = -(target * np.log(y_pred_clip) + (1 - target) * np.log(1 - y_pred_clip))
        sample_loss = np.mean(sample_loss, axis=-1)
        return sample_loss

    def backward(self, dvalues, target):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clip_dvalue = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(target / clip_dvalue - (1 - target) / (1 - clip_dvalue)) / outputs
        self.dinputs = self.dinputs / samples


class Mean_Square_Error(Loss):
    def forward(self, predictions, target):
        sample = np.mean((target - predictions) ** 2, axis=-1)
        return sample

    def backward(self, dvalues, target):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (target - dvalues) / outputs
        self.dinputs = self.dinputs / samples

class Mean_Absolute_Error(Loss):
    def forward(self, predictions, target):
        sample = np.mean(np.abs(target - predictions), axis=-1)
        return sample

    def backward(self, dvalues, target):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = np.sign(target - dvalues) / outputs
        self.dinputs = self.dinputs / samples

class Log_Loss(Loss):
    def loss():
        pass
