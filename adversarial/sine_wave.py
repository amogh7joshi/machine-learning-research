import numpy as np
import matplotlib.pyplot as plt

# A Generative adversarial Network made to create sine & cosine waves.

DIM_Z = 1
DIM_X = 10
ITERATIONS = 50000

GENERATOR_HIDDEN = 10
DISCRIMINATOR_HIDDEN = 10
GENERATOR_STEP = 0.01
DISCRIMINATOR_STEP = 0.01

GRADIENT_CLIP = 0.2
WEIGHT_CLIP = 0.25

REDUCTION = 1e-7

# Get a batch of samples.
def samples(random = True):
   if random:
      x = np.random.uniform(0, 1)
      frequency = np.random.uniform(1.2, 1.5)
      mult = np.random.uniform(0.5, 0.8)
   else:
      x = 0
      frequency = 0.2
      mult = 1
   # Edit np.sin/np.cos for sine/cosine waves.
   return np.array([mult * np.cos(x + frequency * i) for i in range(DIM_X)])

# Activation Functions
class Activation(object):
   def Sigmoid(self, x):
      return 1 / (1 + np.exp(-x))

   def dSigmoid(self, x):
      return self.Sigmoid(x) * (1 - self.Sigmoid(x))

   def ReLU(self, x):
      return np.maximum(x, 0)

   def dReLU(self, x):
      return self.ReLU(x)

   def LeakyReLU(self, x, k = 0.2):
      return np.where(x >= 0, x, x * k)

   def dLeakyReLU(self, x, k = 0.2):
      return np.where(x >= 0, 1, k)

   def tanh(self, x):
      return np.tanh(x)

   def dtanh(self, x):
      return 1 - self.tanh(x) ** 2

activator = Activation()

# Initialize the layer parameters.
def initialize_weights(in_layer, out_layer):
   scale = np.sqrt(2 / (in_layer + out_layer))
   return np.random.uniform(-scale, scale, (in_layer, out_layer))

# Loss Function
class Loss(object):
   def __init__(self):
      self.logit = None
      self.label = None

   def forward(self, logit, label):
      if logit[0, 0] < REDUCTION:
         logit[0, 0] = REDUCTION
      if 1 - logit[0, 0] < REDUCTION:
         logit[0, 0] = 1 - REDUCTION
      self.logit = logit
      self.label = label
      return - (label * np.log(logit) + (1 - label) * np.log(1 - logit))

   def backward(self):
      return (1 - self.label) / (1 - self.logit) - self.label / self.logit

# The Generator Network
class Generator(object):
   def __init__(self):
      self.w1 = initialize_weights(DIM_Z, GENERATOR_HIDDEN)
      self.b1 = initialize_weights(1, GENERATOR_HIDDEN)
      self.x1 = None
      self.w2 = initialize_weights(GENERATOR_HIDDEN, GENERATOR_HIDDEN)
      self.b2 = initialize_weights(1, GENERATOR_HIDDEN)
      self.x2 = None
      self.w3 = initialize_weights(GENERATOR_HIDDEN, DIM_X)
      self.b3 = initialize_weights(1, DIM_X)
      self.x3 = None
      self.x = None
      self.z = None

   def feedforward(self, inputs):
      '''
      A feedforward implementation for the network.
      '''
      self.z = inputs.reshape(1, DIM_Z)
      self.x1 = activator.ReLU(np.matmul(self.z, self.w1) + self.b2)
      self.x2 = activator.ReLU(np.matmul(self.x1, self.w2) + self.b2)
      self.x3 = np.matmul(self.x2, self.w3) + self.b3
      self.x = activator.tanh(self.x3)
      return self.x

   def backpropagate(self, outputs):
      '''
      An implementation of backpropagation for the network.
      '''
      delta = outputs * activator.dtanh(self.x)

      # Third Layer
      dw3 = np.matmul(np.transpose(self.x2), delta)
      db3 = delta.copy()
      delta = np.matmul(delta, np.transpose(self.w3))
      if (np.linalg.norm(dw3) > GRADIENT_CLIP):
         dw3 = GRADIENT_CLIP / np.linalg.norm(dw3) * dw3
      self.w3 -= GENERATOR_STEP * dw3
      self.w3 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w3))
      self.b3 = -GENERATOR_STEP * db3
      self.b3 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b3))
      delta *= activator.dReLU(self.x2)

      # Second Layer
      dw2 = np.matmul(np.transpose(self.x1), delta)
      db2 = delta.copy()
      delta = np.matmul(delta, np.transpose(self.w2))
      if (np.linalg.norm(dw2) > GRADIENT_CLIP):
         dw2 = GRADIENT_CLIP / np.linalg.norm(dw2) * dw2
      self.w2 -= GENERATOR_STEP * dw2
      self.w2 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w2))
      self.b2 = -GENERATOR_STEP * db2
      self.b2 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b2))
      delta *= activator.dReLU(self.x1)

      # First Layer
      dw1 = np.matmul(np.transpose(self.z), delta)
      db1 = delta.copy()
      if (np.linalg.norm(dw1) > GRADIENT_CLIP):
         dw1 = GRADIENT_CLIP / np.linalg.norm(dw1) * dw1
      self.w1 -= GENERATOR_STEP * dw1
      self.w1 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w1))
      self.b1= -GENERATOR_STEP * db1
      self.b1 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b1))

# The Discriminator Network
class Discriminator(object):
   def __init__(self):
      self.w1 = initialize_weights(DIM_X, DISCRIMINATOR_HIDDEN)
      self.b1 = initialize_weights(1, DISCRIMINATOR_HIDDEN)
      self.y1 = None
      self.w2 = initialize_weights(DISCRIMINATOR_HIDDEN, DISCRIMINATOR_HIDDEN)
      self.b2 = initialize_weights(1, DISCRIMINATOR_HIDDEN)
      self.y2 = None
      self.w3 = initialize_weights(DISCRIMINATOR_HIDDEN, 1)
      self.b3 = initialize_weights(1, 1)
      self.y3 = None
      self.y = None
      self.x = None

   def feedforward(self, inputs):
      '''
      A feedforward implementation for the network.
      '''
      self.x = inputs.reshape(1, DIM_X)
      self.y1 = activator.LeakyReLU(np.matmul(self.x, self.w1) + self.b1)
      self.y2 = activator.LeakyReLU(np.matmul(self.y1, self.w2) + self.b2)
      self.y3 = np.matmul(self.y2, self.w3) + self.b3
      self.y = activator.Sigmoid(self.y3)
      return self.y

   def backpropagate(self, outputs, apply_gradients = True):
      '''
      An implementation of backpropagation for the network.
      '''
      delta = outputs * activator.dSigmoid(self.y)

      # Third Layer
      dw3 = np.matmul(np.transpose(self.y2), delta)
      db3 = delta.copy()
      delta = np.matmul(delta, np.transpose(self.w3))
      if apply_gradients:
         if np.linalg.norm(dw3) > GRADIENT_CLIP:
            dw3 = GRADIENT_CLIP / np.linalg.norm(dw3) * dw3
         self.w3 += DISCRIMINATOR_STEP * dw3
         self.w3 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w3))
         self.b3 += DISCRIMINATOR_STEP * db3
         self.b3 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b3))
      delta *= activator.dLeakyReLU(self.y2)

      # Second Layer
      dw2 = np.matmul(np.transpose(self.y1), delta)
      db2 = delta.copy()
      delta = np.matmul(delta, np.transpose(self.w2))
      if apply_gradients:
         if np.linalg.norm(dw2) > GRADIENT_CLIP:
            dw3 = GRADIENT_CLIP / np.linalg.norm(dw2) * dw2
         self.w2 += DISCRIMINATOR_STEP * dw2
         self.w2 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w2))
         self.b2 += DISCRIMINATOR_STEP * db2
         self.b2 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b2))
      delta *= activator.dLeakyReLU(self.y1)

      # First Layer
      dw1 = np.matmul(np.transpose(self.x), delta)
      db1 = delta.copy()
      delta = np.matmul(delta, np.transpose(self.w1))
      if apply_gradients:
         if np.linalg.norm(dw1) > GRADIENT_CLIP:
            dw1 = GRADIENT_CLIP / np.linalg.norm(dw1) * dw1
         self.w1 += DISCRIMINATOR_STEP * dw1
         self.w1 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w1))
         self.b1 += DISCRIMINATOR_STEP * db1
         self.b1 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b1))
      return delta

Generator = Generator()
Discriminator = Discriminator()
loss = Loss()

real = 1
fake = 0

def train(generator, discriminator, iterations):
   for i in range(iterations):
      x_real = samples(True)
      y_real = discriminator.feedforward(x_real)
      loss_D = loss.forward(y_real, real)
      dloss_D = loss.backward()
      discriminator.backpropagate(dloss_D)

      noise_z = np.random.randn(DIM_Z)
      x_fake = generator.feedforward(noise_z)
      y_fake = discriminator.feedforward(x_fake)
      loss_Df = loss.forward(y_fake, fake)
      dloss_D = loss.backward()
      discriminator.backpropagate(dloss_D)

      y_faker = discriminator.feedforward(x_fake)
      loss_G = loss.forward(y_faker, real)
      dloss_G = discriminator.backpropagate(loss_G, apply_gradients = True)
      generator.backpropagate(dloss_G)
      loss_Dt = loss_D + loss_Df
      if i % 100 == 0:
         print('"Iter: {}, {} {} {}'.format(i, loss_D.item((0, 0)), loss_Df.item((0, 0)), loss_G.item((0, 0))))

train(Generator, Discriminator, ITERATIONS)
x = np.linspace(0, 10, 10)
for i in range(50):
   noise = np.random.randn(DIM_Z)
   x_fake = Generator.feedforward(noise)
   plt.plot(x, x_fake.reshape(DIM_X))
plt.ylim((-1, 1))
plt.show()



