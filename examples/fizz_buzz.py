"""
Print the numbers 1 to 100, expect
* if the number is divisible by 3, print "fizz"
* if the number is divisible by 5, print "buzz"
* if the number is divisible by 15, print "fizz_buzz"
"""

import autograd
import autograd.nn.functional as F
from autograd.np import np
from autograd import Tensor, Parameter, Module
from autograd.optim import SGD, Adam
from autograd.utils import to_cpu
from autograd.nn import Linear, Dropout
from typing import List

def binary_encode(x: int) -> List[int]:
    return [x >> i & 1 for i in range(10)]

def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]
x_train = Tensor([binary_encode(x) for x in range(101,1024)])
y_train = Tensor([fizz_buzz_encode(x) for x in range(101,1024)])

class FizzBuzzModel(Module):
    def __init__(self, num_hidden: int = 50) -> None:
        self.fc1 = Linear(10, num_hidden)
        self.fc2 = Linear(num_hidden, 4)
        self.dropout = Dropout(0.1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.predict(inputs)

    def predict(self, inputs: Tensor) -> Tensor:
        # inputs will be (batch_size, 10)
        x1 = self.fc1(inputs) # (batch_size, num_hidden)
        x2 = F.tanh(x1)       # (batch_size, num_hidden)
        x2 = self.dropout(x2)
        x3 = self.fc2(x2)     # (batch_size, 4)

        return x3

batch_size = 32
model = FizzBuzzModel()
print(x_train.shape)
starts = np.arange(0, x_train.shape[0],batch_size)
# optimizer = SGD(model.parameters(), 0.001)
optimizer = Adam(model.parameters(), 0.01)
for epoch in range(2000):
   
    epoch_loss = 0.0

    np.random.shuffle(starts)

    for start in starts:
        end = start + batch_size

        optimizer.zero_grad()
        inputs = x_train[start:end]
        predicted = model(inputs)
        actual = y_train[start:end]

        errors = predicted - actual
        loss = (errors * errors).sum()

        loss.backward()
        epoch_loss += loss.data

        optimizer.step()
        
    print(epoch,epoch_loss)

num_correct = 0
for x in range(1, 101):
    inputs = Tensor([binary_encode(x)])
    predicted = model.predict(inputs) # (1,4)
    # print(predicted)
    predicted_idx = autograd.argmax(predicted, -1)[0]
    actual_idx = autograd.argmax(Tensor(fizz_buzz_encode(x)), -1)
    labels = [str(x), "fizz", "buzz", "fizzbuzz"]
    # print(predicted_idx, actual_idx)
    if predicted_idx == actual_idx:
        num_correct += 1
    print(x, labels[predicted_idx.cpu()],labels[actual_idx.cpu()])

print(num_correct, "/100")
    
