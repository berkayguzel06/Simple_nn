import numpy as np

class Loss:
    def remember_trainable_layers(self, trainable_layers):
            self.trainable_layers = trainable_layers
    def calculate(self, output, y):
        # Calculate and return the mean of losses
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
    # Commonly used with a softmax activation on the output layer.
    def forward(self, prediction, target):
        samples = len(prediction)
        # Clip predicted values to avoid log(0) issues
        y_clipped = np.clip(prediction, 1e-7, 1 - 1e-7)
        if len(target.shape) == 1:
            # Calculate negative log likelihood for each correct prediction
            correctConfidences = y_clipped[range(samples), target]
        elif len(target.shape) == 2:
            # Calculate negative log likelihood for each example in the batch
            correctConfidences = np.sum(y_clipped * target, axis=1)
        log = -np.log(correctConfidences)
        return log
    
    def backward(self, dvalues, target):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(target.shape) == 1:
            # Convert target to one-hot encoding
            target = np.eye(labels)[target]

        # Calculate gradient of the loss with respect to predicted values
        self.dinputs = -target / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy(Loss):
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Binary_Cross_Entropy(Loss):
    def forward(self, predictions, target):
        # Clip predicted values to avoid log(0) issues
        y_pred_clip = np.clip(predictions, 1e-7, 1 - 1e-7)
        # Calculate and return the mean of binary cross-entropy losses
        sample_loss = -(target * np.log(y_pred_clip) + (1 - target) * np.log(1 - y_pred_clip))
        sample_loss = np.mean(sample_loss, axis=-1)
        return sample_loss

    def backward(self, dvalues, target):
        # Number of samples
        samples = len(dvalues)
        # Number of output values
        outputs = len(dvalues[0])
        # Clip values to avoid division by zero issues
        clip_dvalue = np.clip(dvalues, 1e-7, 1 - 1e-7)
        # Calculate gradient of the loss with respect to predicted values
        self.dinputs = -(target / clip_dvalue - (1 - target) / (1 - clip_dvalue)) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples


class Mean_Square_Error(Loss):
    def forward(self, predictions, target):
        # Calculate and return the mean of squared errors
        sample = np.mean((target - predictions) ** 2, axis=-1)
        return sample

    def backward(self, dvalues, target):
        # Number of samples
        samples = len(dvalues)
        # Number of output values
        outputs = len(dvalues[0])
        # Calculate gradient of the loss with respect to predicted values
        self.dinputs = -2 * (target - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Mean_Absolute_Error(Loss):
    def forward(self, predictions, target):
        # Calculate and return the mean of absolute errors
        sample = np.mean(np.abs(target - predictions), axis=-1)
        return sample

    def backward(self, dvalues, target):
        # Number of samples
        samples = len(dvalues)
        # Number of output values
        outputs = len(dvalues[0])
        # Calculate gradient of the loss with respect to predicted values
        self.dinputs = np.sign(target - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples



class Log_Loss(Loss):
    # Commonly used for binary classification (0 or 1)
    def loss():
        pass
