import flax.linen as nn
import jax
import jax.numpy as jnp
# Model
class Model(nn.Module):
  @nn.compact
  def __call__(self,inputs):
    x=nn.Dense(256)(inputs)
    x=nn.selu(x)
    x=nn.Dense(4)(inputs)
    x=nn.selu(x)
    x=nn.Dense(4)(x)
    x=nn.selu(x)
    x=nn.Dense(3)(x)
    x=nn.softmax(x)
    return x

if __name__ == "__main__":
  model=Model
  model()
