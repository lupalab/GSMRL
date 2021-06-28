import tensorflow as tf
import numpy as np


def dense_nn(inputs, layer_sizes, name):
    output = inputs
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        i = -1
        for i, size in enumerate(layer_sizes[:-1]):
            output = tf.layers.dense(output, size, name=f'layer_{i}')
            output = tf.nn.relu(output)
        output = tf.layers.dense(output, layer_sizes[-1], name=f'layer_{i+1}')

    return output

def deepset(inputs, layer_sizes, name, mask=None):
    output = inputs
    if mask is None:
        mask = tf.ones([tf.shape(inputs)[0], tf.shape(inputs)[1], 1], dtype=tf.float32)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        i = -1
        for i, size in enumerate(layer_sizes[:-1]):
            output = tf.layers.dense(output, size, name=f'layer_{i}')
            output = tf.nn.elu(output)
            output = output - tf.reduce_mean(output * mask, axis=1, keepdims=True)
        output = tf.layers.dense(output, layer_sizes[-1], name=f'layer_{i+1}')

    return output

def set_attention(Q, K, dim, num_heads, name='set_attention', mask=None):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        q = tf.layers.dense(Q, dim, name='query')
        k = tf.layers.dense(K, dim, name='key')
        v = tf.layers.dense(K, dim, name='value')

        q_ = tf.concat(tf.split(q, num_heads, axis=-1), axis=0)
        k_ = tf.concat(tf.split(k, num_heads, axis=-1), axis=0)
        v_ = tf.concat(tf.split(v, num_heads, axis=-1), axis=0)

        logits = tf.matmul(q_, k_, transpose_b=True)/np.sqrt(dim) # [B*Nh,Nq,Nk]
        inf_logits = -tf.ones_like(logits) * np.inf
        if mask is None:
            mask = tf.ones_like(logits)
        else:
            mask = tf.tile(tf.transpose(mask, [0,2,1]), [num_heads, tf.shape(logits)[1],1])
        logits = tf.where(tf.equal(mask, 0), inf_logits, logits)

        A = tf.nn.softmax(logits, axis=-1)
        o = q_ + tf.matmul(A, v_)
        o = tf.concat(tf.split(o, num_heads, axis=0), axis=-1)
        o = o + tf.layers.dense(o, dim, activation=tf.nn.relu, name='output')

    return o

def set_transformer(inputs, layer_sizes, name, num_heads=4, num_inds=16, mask=None):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        out = inputs
        for i, size in enumerate(layer_sizes):
            inds = tf.get_variable(f'inds_{i}', shape=[1,num_inds,size], dtype=tf.float32, trainable=True,
                                    initializer=tf.contrib.layers.xavier_initializer())
            inds = tf.tile(inds, [tf.shape(out)[0],1,1])
            tmp = set_attention(inds, out, size, num_heads, name=f'self_attn_{i}_pre', mask=mask)
            out = set_attention(out, tmp, size, num_heads, name=f'self_attn_{i}_post', mask=None)

        seed = tf.get_variable('pool_seed', shape=[1,1,size], dtype=tf.float32, trainable=True, 
                                initializer=tf.contrib.layers.xavier_initializer())
        seed = tf.tile(seed, [tf.shape(out)[0],1,1])
        out = set_attention(seed, out, size, num_heads, name='pool_attn', mask=mask)
        out = tf.squeeze(out, axis=1)
        out = tf.layers.dense(out, size, name='output')

    return out

def uniform_attention(q, v, mask):
    """Uniform attention. Equivalent to np.

    Args:
      q: queries. tensor of shape [B,m,d_k].
      v: values. tensor of shape [B,n,d_v].
      mask: [B,n,1]

    Returns:
      tensor of shape [B,m,d_v].
    """
    total_points = tf.shape(q)[1]
    rep = tf.reduce_mean(v * mask, axis=1, keepdims=True)  # [B,1,d_v]
    rep = tf.tile(rep, [1, total_points, 1])
    return rep


def laplace_attention(q, k, v, scale, normalise, mask):
    """Computes laplace exponential attention.

    Args:
      q: queries. tensor of shape [B,m,d_k].
      k: keys. tensor of shape [B,n,d_k].
      v: values. tensor of shape [B,n,d_v].
      scale: float that scales the L1 distance.
      normalise: Boolean that determines whether weights sum to 1.
      mask: [B,n,1]

    Returns:
      tensor of shape [B,m,d_v].
    """
    k = tf.expand_dims(k, axis=1)  # [B,1,n,d_k]
    q = tf.expand_dims(q, axis=2)  # [B,m,1,d_k]
    unnorm_weights = - tf.abs((k - q) / scale)  # [B,m,n,d_k]
    unnorm_weights = tf.reduce_sum(unnorm_weights, axis=-1)  # [B,m,n]
    mask = tf.tile(tf.transpose(mask, [0,2,1]), [1,tf.shape(q)[1],1]) # [B,m,n]
    inf_weights = -tf.ones_like(unnorm_weights) * np.inf
    unnorm_weights = tf.where(tf.equal(mask, 0), inf_weights, unnorm_weights)
    if normalise:
        weight_fn = tf.nn.softmax
    else:
        def weight_fn(x): return 1 + tf.tanh(x)
    weights = weight_fn(unnorm_weights)  # [B,m,n]
    rep = tf.einsum('bik,bkj->bij', weights, v)  # [B,m,d_v]
    return rep


def dot_product_attention(q, k, v, normalise, mask):
    """Computes dot product attention.

    Args:
      q: queries. tensor of  shape [B,m,d_k].
      k: keys. tensor of shape [B,n,d_k].
      v: values. tensor of shape [B,n,d_v].
      normalise: Boolean that determines whether weights sum to 1.
      mask: [B,n,1]

    Returns:
      tensor of shape [B,m,d_v].
    """
    d_k = tf.shape(q)[-1]
    scale = tf.sqrt(tf.cast(d_k, tf.float32))
    unnorm_weights = tf.einsum('bjk,bik->bij', k, q) / scale  # [B,m,n]
    mask = tf.tile(tf.transpose(mask, [0,2,1]), [1,tf.shape(q)[1],1]) # [B,m,n]
    inf_weights = -tf.ones_like(unnorm_weights) * np.inf
    unnorm_weights = tf.where(tf.equal(mask, 0), inf_weights, unnorm_weights)
    if normalise:
        weight_fn = tf.nn.softmax
    else:
        weight_fn = tf.sigmoid
    weights = weight_fn(unnorm_weights)  # [B,m,n]
    rep = tf.einsum('bik,bkj->bij', weights, v)  # [B,m,d_v]
    return rep


def multihead_attention(q, k, v, num_heads, mask):
    """Computes multi-head attention.

    Args:
      q: queries. tensor of  shape [B,m,d_k].
      k: keys. tensor of shape [B,n,d_k].
      v: values. tensor of shape [B,n,d_v].
      num_heads: number of heads. Should divide d_v.

    Returns:
      tensor of shape [B,m,d_v].
    """
    d_k = q.get_shape().as_list()[-1]
    d_v = v.get_shape().as_list()[-1]
    head_size = d_v / num_heads
    key_initializer = tf.random_normal_initializer(stddev=d_k**-0.5)
    value_initializer = tf.random_normal_initializer(stddev=d_v**-0.5)
    rep = tf.constant(0.0)
    for h in range(num_heads):
        o = dot_product_attention(
            tf.layers.conv1d(q, head_size, 1, kernel_initializer=key_initializer,
                             name=f'wq_{h}', use_bias=False, padding='valid'),
            tf.layers.conv1d(k, head_size, 1, kernel_initializer=key_initializer,
                             name=f'wk_{h}', use_bias=False, padding='valid'),
            tf.layers.conv1d(v, head_size, 1, kernel_initializer=key_initializer,
                             name=f'wv_{h}', use_bias=False, padding='valid'),
            normalise=True, mask=mask)
        rep += tf.layers.conv1d(o, d_v, 1, kernel_initializer=value_initializer,
                                name=f'wo_{h}', use_bias=False, padding='valid')
    return rep


class Attention(object):
    """The Attention module."""

    def __init__(self, rep, output_sizes, att_type, scale=1., normalise=True, num_heads=8, name='attention'):
        """Create attention module.

        Takes in context inputs, target inputs and
        representations of each context input/output pair
        to output an aggregated representation of the context data.
        Args:
          rep: transformation to apply to contexts before computing attention. 
              One of: ['identity','mlp'].
          output_sizes: list of number of hidden units per layer of mlp.
              Used only if rep == 'mlp'.
          att_type: type of attention. One of the following:
              ['uniform','laplace','dot_product','multihead']
          scale: scale of attention.
          normalise: Boolean determining whether to:
              1. apply softmax to weights so that they sum to 1 across context pts or
              2. apply custom transformation to have weights in [0,1].
          num_heads: number of heads for multihead.
        """
        self._rep = rep
        self._output_sizes = output_sizes
        self._type = att_type
        self._scale = scale
        self._normalise = normalise
        if self._type == 'multihead':
            self._num_heads = num_heads
        self._name = name

    def __call__(self, x1, x2, r, mask=None):
        """Apply attention to create aggregated representation of r.

        Args:
        x1: tensor of shape [B,n2,d_x].
        x2: tensor of shape [B,n1,d_x].
        r: tensor of shape [B,n1,d].
        
        Returns:
        tensor of shape [B,n2,d]

        Raises:
        NameError: The argument for rep/type was invalid.
        """
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            if self._rep == 'identity':
                q, k = (x1, x2)
            elif self._rep == 'mlp':
                # Pass through MLP
                q = dense_nn(x1, self._output_sizes, "attention")
                k = dense_nn(x2, self._output_sizes, "attention")
            else:
                raise NameError("'rep' not among ['identity','mlp']")

            if mask is None:
                mask = tf.ones([tf.shape(r)[0], tf.shape[1], 1], dtype=tf.float32)

            if self._type == 'uniform':
                rep = uniform_attention(q, r, mask)
            elif self._type == 'laplace':
                rep = laplace_attention(q, k, r, self._scale, self._normalise, mask)
            elif self._type == 'dot_product':
                rep = dot_product_attention(q, k, r, self._normalise, mask)
            elif self._type == 'multihead':
                rep = multihead_attention(q, k, r, self._num_heads, mask)
            else:
                raise NameError(("'att_type' not among ['uniform','laplace','dot_product','multihead']"))

        return rep
