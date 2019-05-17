from numpy import random, dot, array, exp
class NeuralNetwork():
  def __init__(self):
    random.seed(1)
    self.synaptic_weights = 2 * random.random((2, 1)) - 1
  def train(self, inputs, outputs, iterations):
    for i in range(iterations):
      output = self.think(inputs)
      error = outputs - output
      adjustment = dot(inputs.T, error * self.__sigmoid_derivative(output))
      self.synaptic_weights += adjustment
  def think(self, inputs):
    return(self.__sigmoid(dot(inputs, self.synaptic_weights)))
  def __sigmoid(self, x):
    return(1/(1+exp(-x)))
  def __sigmoid_derivative(self, x):
    return(x*(1-x))
if __name__ == "__main__":
  nn = NeuralNetwork()
  #First item in inputs array is weight in grams, second is color of fruit (green = 0; yellow = 1)
  inputs = array([[3, 0], [5, 0], [9, 1], [11, 1], [12, 1]])
  #0 = Lime; 1 = Lemon
  outputs = array([[0, 0, 1, 1, 1]]).T
  nn.train(inputs, outputs, 10000)
  print("New Weights:")
  print(nn.synaptic_weights)
  print("Enter New Weight:")
  x = int(input(">> "))
  print("Enter Color (green(0)/yellow(1)):")
  y = int(input(">> "))
  z = [x, y]
if nn.think(array(z)) < 0.5:
  print("Fruit is a Lime")
elif nn.think(array(z)) >= 0.5:
  print("Fruit is a Lemon")