import tensorflow as tf
from tensorflow.keras import layers

latent_dim = 2

# Encoder
encoder = tf.keras.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(64, activation='relu'),
    layers.Dense(latent_dim)
])

# Decoder
decoder = tf.keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(28*28, activation='sigmoid'),
    layers.Reshape((28,28))
])

# VAE Forward
def vae(x):
    z = encoder(x)
    reconstructed = decoder(z)
    return reconstructed

# Test
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0

output = vae(x_train[:1])
print("Output shape:", output.shape)
