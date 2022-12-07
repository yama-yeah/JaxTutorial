import jax.numpy as jnp
import jax
import numpy as np
@jax.jit
def rmse(y_true,y_pred):
  y_pred=jnp.ravel(y_pred)
  print(y_pred)
  print(y_true)
  return jnp.mean((y_true-y_pred)**2)**0.5
@jax.jit
def accuracy(y_true,y_pred):
  #y_predを平たくする
  y_pred=jnp.ravel(y_pred)
  return jnp.sum(y_true==y_pred)/y_true.shape[0]
@jax.jit
def categorical_crossentropy(y_true,y_pred):
  y_pred= y_pred / jnp.sum(y_pred, axis=-1, keepdims=True)
  return -jnp.sum(y_true*jnp.log(y_pred))