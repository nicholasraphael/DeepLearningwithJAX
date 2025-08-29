#%%
# einops


# Compute math

# Matrix muls dominate the compute -> 2 * m * n * p FLOPs
# FLOP/s depends on hardware and data types

# num_of_forward_flops = (2 * params_1) + (2 * params_2) + ...
# num_of_backward_flops = 4 * params



# A simple example of 2 layer NN in JAX for MINST
import time
from jax import random, vmap, value_and_grad, jit
import jax.numpy as jnp
from jax.nn import swish, logsumexp, one_hot

LAYER_SIZES = [28*28, 512, 10]
PARAM_SCALE = 0.01

num_epochs = 25

def init_network_parameters(sizes, key=random.PRNGKey(0), scale=1e-2):
  """Initialize all layers for a fully-connected neural network with given sizes"""

  def random_layer_params(m, n, key, scale=1e-2):
    """helper to randomly initialize W and b of a dense layer"""
    W_key, b_key = random.split(key=key)
    return scale * random.normal(W_key, (n, m)), scale * random.normal(b_key, (n,))
  
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k, scale) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


params = init_network_parameters(LAYER_SIZES, random.PRNGKey(0), scale=PARAM_SCALE)

def call(params, image):
  """Forawd pass"""
  activations = image
  for w, b in params[:-1]:
    outputs = jnp.dot(w, activations) + b
    activations = swish(outputs)
  
  # for the last layer we don not apply the activation function
  final_w, final_b = params[-1]
  logits = jnp.dot(final_w, activations) + final_b
  return logits

# a version that can now work with batches
batched_call = vmap(call, in_axes=(None, 0))


def loss(params, images, targets):
  """Categorical cross entropy loss function"""
  logits = batched_call(params, images)
  log_preds = logits - logsumexp(logits)
  return -jnp.mean(targets*log_preds)



INIT_LR = 1.0
DECAY_RATE = 0.95
DECAY_STEPS = 5


@jit
def update(params, x, y, epoch_number):
  loss_value, grads = value_and_grad(loss)(params, x, y)

  lr = INIT_LR * DECAY_RATE ** (epoch_number / DECAY_STEPS)
  return [(w - lr * dw , b - lr * db) for (w, b), (dw, db) in zip(params, grads)], loss_value

#%%
import jax
import jax.numpy as jnp

# placing data to a device

arr = jnp.array([1 , 2, 3])
arr.device

arr_cpu = jax.device_put(arr, jax.devices('cpu')[0])

arr + arr_cpu

arr_gpu = jax.device_put(arr, jax.devices('gpu')[0])

try:
  arr_gpu + arr_cpu
except ValueError as e:
  print(e)


# %%
# Working with Async dispatch
import jax
import jax.numpy as jnp

a = jnp.array(range(1_000_000)).reshape(1000, 1000)
a.device
# only measure time to dispatch the work
%time x = jnp.dot(a, a) 

%time x = jnp.dot(a,a).block_until_ready()
# %%
# updating a tensor element
import jax.numpy as jnp 
a_jnp = jnp.array([1,2,4])

a_jnp = a_jnp.at[2].set(3)

a_jnp[2]

a_jnp  = jnp.array(range(10))
a_jnp.at[42].get(mode='drop')

# %%
# working with autodiff

def f(x):
  return x**4 + 12*x + 1/x

x = 11.0

df = jax.grad(f)
print(df(x))
# %%

# Linear regression example with autodiff
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10*np.pi, num=1000)
e = np.random.normal(scale=10.0, size=x.size)
y = 65.0 + 1.8*x + 40*np.cos(x) + e

# plt.scatter(x, y)

xt = jnp.array(x)
yt = jnp.array(y)

learning_rate = 1e-2

# W, b
model_params = jnp.array([1. , 1.])

def model(theta ,x):
  w, b = theta
  return w * x + b


def loss_fn(model_params, x, y,):
  prediction = model(model_params, x)
  return jnp.mean((prediction - y)**2), prediction

# creates a function for calc gradients 
# jax calculates gradients with respect to the first parameter of a function
# see jax.grad(f, argnums=(1,..)) otherwise
grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

# calc the actual gradients
(loss, preds), gradients = grad_fn(model_params, xt, yt)

print(loss, gradients)

# update step
model_params -= learning_rate * gradients
#%%
# Jacobians
import jax
import jax.numpy as jnp

def f(x):
  return [
    x[0]**2 + x[1]**2 - x[1]*x[2],
    x[0]**2 - x[1]**2 + 3*x[0]*x[2]
  ]

print(jax.jacrev(f) (jnp.array([3., 4., 5.])))
