import numpy as np
import nnfs
from nnfs.datasets import sine_data
import matplotlib.pyplot as plt
nnfs.init()
#密集层类
class DenseLayer:
    #初始化权重与偏置
    def __init__(self, n_inputs, n_neurons,weight_regularizer_l1=0,weight_regularizer_l2=0,bias_regularizer_l1=0,bias_regularizer_l2=0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1=weight_regularizer_l1
        self.weight_regularizer_l2=weight_regularizer_l2
        self.bias_regularizer_l1=bias_regularizer_l1
        self.bias_regularizer_l2=bias_regularizer_l2
        # self.dweights=None
        # self.dbiases=None
    #向前传播
    def forward(self, inputs):
        self.inputs=inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regularizer_l1>0:
            dL1=np.ones_like(self.dweights)
            dL1[self.dweights<0]=-1
            self.dweights+=self.weight_regularizer_l1*dL1
        if self.weight_regularizer_l2>0:
            self.dweights+=self.weight_regularizer_l2*2*self.weights

        if self.bias_regularizer_l1>0:
            dL1=np.ones_like(self.dbiases)
            dL1[self.dbiases<0]=-1
            self.dbiases+=self.bias_regularizer_l1*dL1
        if self.bias_regularizer_l2>0:
            self.dbiases+=self.bias_regularizer_l2*2*self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)

class Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate
    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape)/self.rate
        self.output = inputs * self.binary_mask
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

#激活函数类
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs=inputs
        self.output = np.maximum(0, inputs)

    def backward(self,dvalues):
        self.dinputs=dvalues.copy()
        self.dinputs[self.inputs<0]=0




class Activation_Softmax:
    def forward(self, inputs):
        self.inputs=inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self,dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalue) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalue)

class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs=inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self,dvalues):
        self.dinputs=dvalues*(1-self.output)*self.output

class Activation_Softplus:
    def forward(self, inputs):
        self.inputs=inputs
        self.output = np.log(1 + np.exp(inputs))
    def backward(self,dvalues):
        self.dinputs=dvalues*(1-self.output)/(1+np.exp(-self.inputs))

class Activation_LeakyReLU:
    def __init__(self,alpha=0.01):
        self.alpha=alpha
    def forward(self, inputs):
        self.inputs=inputs
        self.output=np.maximum(self.alpha*inputs,inputs)
    def backward(self,dvalues):
        self.dinputs=dvalues.copy()
        self.dinputs[self.inputs<0]=self.alpha

class Activation_Linear:
    def forward(self, inputs):
        self.inputs=inputs
        self.output=inputs
    def backward(self,dvalues):
        self.dinputs=dvalues.copy()

class Loss:
    def calculate(self, output, y):#计算损失函数
        sample_losses = self.forward(output, y)#计算每个样本的损失
        data_loss = np.mean(sample_losses)#计算平均损失
        return data_loss
    def regularization_loss(self,layer):
        regularization_loss=0

        if layer.weight_regularizer_l1>0:
            regularization_loss+=layer.weight_regularizer_l1*np.sum(np.abs(layer.weights))
        if layer.weight_regularizer_l2>0:
            regularization_loss+=layer.weight_regularizer_l2*np.sum(layer.weights**2)
        if layer.bias_regularizer_l1>0:
            regularization_loss+=layer.bias_regularizer_l1*np.sum(np.abs(layer.biases))
        if layer.bias_regularizer_l2>0:
            regularization_loss+=layer.bias_regularizer_l2*np.sum(layer.biases**2)
        return regularization_loss

class Loss_CategoricalCrossentropy(Loss):#交叉熵损失函数
    def forward(self, y_pred, y_true):#计算每个样本的损失
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)#防止log(0),clip()函数将y_pred的值限制在1e-7和1-1e-7之间
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class Loss_BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses
    def backward(self,dvalues,y_true):
        samples=len(dvalues)
        outputs=len(dvalues[0])
        clipped_dvalues=np.clip(dvalues,1e-7,1-1e-7)
        self.dinputs=-(y_true/clipped_dvalues-(1-y_true)/(1-clipped_dvalues))/outputs
        self.dinputs=self.dinputs/samples
        #return self.dinputs

class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)
        return sample_losses
    def backward(self,dvalues,y_true):
        samples=len(dvalues)
        outputs=len(dvalues[0])
        self.dinputs=-2*(y_true-dvalues)/outputs
        self.dinputs=self.dinputs/samples
        return self.dinputs

class Loss_MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses
    def backward(self,dvalues,y_true):
        samples=len(dvalues)
        outputs=len(dvalues[0])
        self.dinputs=np.sign(y_true-dvalues)/outputs
        self.dinputs=self.dinputs/samples
        return self.dinputs

class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self,inputs,y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self,dvalues,y_true):
        samples=len(dvalues)
        if y_true.shape==2:
            y_true=np.argmax(y_true,axis=1)
        self.dinputs=dvalues.copy()
        self.dinputs[range(samples),y_true]-=1
        self.dinputs=self.dinputs/samples
        return self.dinputs

class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        self.velocity = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self,layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums=np.zeros_like(layer.weights)
                layer.bias_momentums=np.zeros_like(layer.biases)
            weight_updates=self.momentum*layer.weight_momentums-self.current_learning_rate*layer.dweights
            bias_updates=self.momentum*layer.bias_momentums-self.current_learning_rate*layer.dbiases
            layer.weight_momentums=weight_updates
            layer.bias_momentums=bias_updates
        else:
            weight_updates=-self.current_learning_rate*layer.dweights
            bias_updates=-self.current_learning_rate*layer.dbiases
        layer.weights+=weight_updates
        layer.biases+=bias_updates

    def post_update_params(self):
        self.iterations+=1

class Optimizer_Adam:
    def __init__(self,learning_rate=0.001,decay=0.0,epsilon=1e-7,beta_1=0.9,beta_2=0.999):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.iterations=0
        self.epsilon=epsilon
        self.beta_1=beta_1
        self.beta_2=beta_2
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate*(1.0/(1.0+self.decay*self.iterations))
    def update_params(self,layer):
        if not hasattr(layer,'weight_cache'):
            layer.weight_momentums=np.zeros_like(layer.weights)
            layer.weight_cache=np.zeros_like(layer.weights)
            layer.bias_momentums=np.zeros_like(layer.biases)
            layer.bias_cache=np.zeros_like(layer.biases)
        layer.weight_momentums=self.beta_1*layer.weight_momentums+(1-self.beta_1)*layer.dweights
        layer.bias_momentums=self.beta_1*layer.bias_momentums+(1-self.beta_1)*layer.dbiases
        weight_momentums_corrected=layer.weight_momentums/(1-self.beta_1**(self.iterations+1))
        bias_momentums_corrected=layer.bias_momentums/(1-self.beta_1**(self.iterations+1))
        layer.weight_cache=self.beta_2*layer.weight_cache+(1-self.beta_2)*layer.dweights**2
        layer.bias_cache=self.beta_2*layer.bias_cache+(1-self.beta_2)*layer.dbiases**2
        weight_cache_corrected=layer.weight_cache/(1-self.beta_2**(self.iterations+1))
        bias_cache_corrected=layer.bias_cache/(1-self.beta_2**(self.iterations+1))
        weight_updates=weight_momentums_corrected/(np.sqrt(weight_cache_corrected)+self.epsilon)
        bias_updates=bias_momentums_corrected/(np.sqrt(bias_cache_corrected)+self.epsilon)
        layer.weights+=-self.current_learning_rate*weight_updates
        layer.biases+=-self.current_learning_rate*bias_updates
    def post_update_params(self):
        self.iterations+=1

class Optimizer_RMSprop:
    def __init__(self,learning_rate=0.001,decay=0.0,epsilon=1e-7,rho=0.9):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.iterations=0
        self.epsilon=epsilon
        self.rho=rho
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate*(1.0/(1.0+self.decay*self.iterations))
    def update_params(self,layer):
        if not hasattr(layer,'weight_cache'):
            layer.weight_cache=np.zeros_like(layer.weights)
            layer.bias_cache=np.zeros_like(layer.biases)
        layer.weight_cache=self.rho*layer.weight_cache+(1-self.rho)*layer.dweights**2
        layer.bias_cache=self.rho*layer.bias_cache+(1-self.rho)*layer.dbiases**2
        weight_updates=self.current_learning_rate*layer.dweights/(np.sqrt(layer.weight_cache)+self.epsilon)
        bias_updates=self.current_learning_rate*layer.dbiases/(np.sqrt(layer.bias_cache)+self.epsilon)
        layer.weights+=-weight_updates
        layer.biases+=-bias_updates
    def post_update_params(self):
        self.iterations+=1

# x,y=spiral_data(samples=100,classes=2)
# y=y.reshape(-1,1)
#
# layer1=DenseLayer(2,64,weight_regularizer_l1=5e-4,weight_regularizer_l2=5e-4)
#
# activation1=Activation_ReLU()
#
# #dropout1=Dropout(0.1)
#
# layer2=DenseLayer(64,1)
# #loss_activation=Activation_Softmax_Loss_CategoricalCrossentropy()
# activation2=Activation_Sigmoid()
# loss_activation=Loss_BinaryCrossentropy()
# optimizer=Optimizer_Adam(decay=5e-7)
#
# for epoch in range(10001):
#     layer1.forward(x)
#     activation1.forward(layer1.output)
#     layer2.forward(activation1.output)
#     activation2.forward(layer2.output)
#
#     loss=loss_activation.calculate(activation2.output,y)
#
#     regulation_loss=loss_activation.regularization_loss(layer1)+loss_activation.regularization_loss(layer2)
#     loss+=regulation_loss
#
#     predictions=(activation2.output>0.5)*1
#
#     accuracy=np.mean(predictions==y)
#     if not epoch%100:
#         print(f"epoch:{epoch}, loss:{loss:.3f}, accuracy:{accuracy:.5f},regulation_loss:{regulation_loss:.5f},"
#               f"learning_rate:{optimizer.current_learning_rate}")
#     loss_activation.backward(activation2.output,y)
#     activation2.backward(loss_activation.dinputs)
#     layer2.backward(activation2.dinputs)
#     #dropout1.backward(layer2.dinputs)
#     activation1.backward(layer2.dinputs)
#     layer1.backward(activation1.dinputs)
#
#     optimizer.pre_update_params()
#     optimizer.update_params(layer1)
#     optimizer.update_params(layer2)
#     optimizer.post_update_params()
#
# x_test,y_test=spiral_data(samples=100,classes=2)
# y_test=y_test.reshape(-1,1)
#
# layer1.forward(x_test)
# activation1.forward(layer1.output)
# layer2.forward(activation1.output)
# activation2.forward(layer2.output)
# loss=loss_activation.calculate(activation2.output,y_test)
# # predictions=np.argmax(loss_activation.output, axis=1)
# # if len(y_test.shape)==2:
# #     y_test=np.argmax(y_test,axis=1)
# predictions=(activation2.output>0.5)*1
# accuracy=np.mean(predictions==y_test)
# print(f"validation, accuracy:{accuracy:.3f},loss:{loss:.3f}")

x,y=sine_data()
dense1=DenseLayer(1,64)
activation1=Activation_ReLU()
dense2=DenseLayer(64,64)
activation2=Activation_ReLU()
dense3=DenseLayer(64,1)
activation3=Activation_Linear()
loss_function=Loss_MeanSquaredError()
optimizer=Optimizer_Adam(learning_rate=0.005,decay=1e-3)

accuracy_precision=np.std(y)/250

for epoch in range(10001):
    dense1.forward(x)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    loss=loss_function.calculate(activation3.output,y)

    reg_loss=loss_function.regularization_loss(dense1)+loss_function.regularization_loss(dense2)+loss_function.regularization_loss(dense3)
    loss+=reg_loss

    predictions=activation3.output
    accuracy=np.mean(np.abs(predictions-y)<accuracy_precision)
    if not epoch%100:
        print(f"epoch:{epoch}, loss:{loss:.3f}, accuracy:{accuracy:.2f}")
    loss_function.backward(activation3.output,y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()

x_test,y_test=sine_data()
dense1.forward(x_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)
loss=loss_function.calculate(activation3.output,y_test)
predictions=activation3.output
accuracy=np.mean(np.abs(predictions-y_test)<accuracy_precision)
print(f"validation, accuracy:{accuracy:.2f},loss:{loss:.3f}")

plt.plot(x_test,y_test)
plt.plot(x_test,predictions)
plt.show()

