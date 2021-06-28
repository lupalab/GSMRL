import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from .utils import dense_nn, set_transformer, Attention

class DeterministicEncoder(object):
    def __init__(self, layer_sizes, attention, name='deterministic_encoder'):
        self._layer_sizes = layer_sizes
        self._attention = Attention('mlp', [layer_sizes[-1]], attention, name='cross_attention')
        self._name = name

    def __call__(self, context_x, context_y, context_m, target_x):
        '''
        Args:
            context_x: [B,Nc,dx]
            context_y: [B,Nc,dy]
            context_m: [B,Nc,1]
            target_x: [B,Nt,dx]
        Returns:
            rc: [B,Nt,d]
        '''
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            inputs = tf.concat([context_x, context_y], axis=-1)
            hidden = dense_nn(inputs, self._layer_sizes, name='nn')
            hidden = self._attention(target_x, context_x, hidden, mask=context_m)

        return hidden

class LatentEncoder(object):
    def __init__(self, layer_sizes, num_latents, num_comps=1, name='latent_encoder'):
        self._layer_sizes = layer_sizes
        self._num_latents = num_latents
        self._num_comps = num_comps
        self._name = name

    def __call__(self, x, y, m, lab=None):
        '''
        Args:
            x: [B,N,dx]
            y: [B,N,dy]
            m: [B,N,1]
            lab: [B,C]
        Returns:
            prior/posterior distribution
        '''
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            inputs = tf.concat([x,y], axis=-1)
            if lab is not None:
                lab = tf.tile(tf.expand_dims(lab, axis=1), [1,tf.shape(x)[1],1])
                inputs = tf.concat([inputs, lab], axis=-1)
            hidden = set_transformer(inputs, self._layer_sizes, name='nn', mask=m)

            if self._num_comps == 1:
                # mean and variance
                layer_sizes = [(self._layer_sizes[-1] + self._num_latents)//2, self._num_latents*2]
                ms = dense_nn(hidden, layer_sizes, name='ms')
                mean, logs = tf.split(ms, 2, axis=1)
                sigma = 0.1 + 0.9 * tf.sigmoid(logs)
                dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=sigma)
            else:
                layer_sizes = [(self._layer_sizes[-1] + self._num_latents*self._num_comps)//2, self._num_latents*self._num_comps]
                mean = dense_nn(hidden, layer_sizes, name='mean')
                mean = tf.reshape(mean, [tf.shape(mean)[0], self._num_comps, self._num_latents])
                logs = dense_nn(hidden, layer_sizes, name='sigma')
                logs = tf.reshape(logs, [tf.shape(logs)[0], self._num_comps, self._num_latents])
                layer_sizes = [(self._layer_sizes[-1] + self._num_comps)//2, self._num_comps]
                logits = dense_nn(hidden, layer_sizes, name='logits')
                dist = tfd.MixtureSameFamily(
                    components_distribution=tfd.MultivariateNormalDiag(
                        loc=mean, scale_diag=tf.nn.softplus(logs)),
                    mixture_distribution=tfd.Categorical(logits=logits),
                    name="latent_dist")

        return dist

class Decoder(object):
    def __init__(self, layer_sizes, name='decoder'):
        self._layer_sizes = layer_sizes
        self._name = name

    def __call__(self, representation, target_x):
        '''
        Args:
            representation: [B,Nt,d]
            target_x: [B,Nt,dx]
        Returns:
            observation distribution
        '''
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            hidden = tf.concat([representation, target_x], axis=-1)
            hidden = dense_nn(hidden, self._layer_sizes, name='nn')
            mean, logs = tf.split(hidden, 2, axis=-1)
            sigma = 0.1 + 0.9 * tf.nn.softplus(logs)
            dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=sigma)

        return dist

class Classifier(object):
    def __init__(self, layer_sizes, name='classifier'):
        self._layer_sizes = layer_sizes
        self._name = name

    def __call__(self, representation):
        '''
        Args:
            representation: [B,d]
        Return:
            logits
        '''
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            logits = dense_nn(representation, self._layer_sizes, name='nn')

        return logits