import tensorflow as tf
import numpy as np


def deepset(inputs, layers_sizes, name="deep_set", reuse=False, mask=None):
    print("Building deep set {} | sizes: {}".format(name, [inputs.shape[-1]] + layers_sizes))

    if mask is None:
        mask = tf.ones([tf.shape(inputs)[0], tf.shape(inputs)[1], 1], dtype=tf.float32)

    with tf.variable_scope(name, reuse=reuse):
        out = inputs
        for i, size in enumerate(layers_sizes):
            print(f"Layer: peq_{i}: {size}")
            out = tf.layers.dense(out, size, name=f'peq_{i}', reuse=reuse)
            out = tf.nn.elu(out)
            out = out - tf.reduce_mean(out*mask, axis=1, keepdims=True)
        out = tf.reduce_mean(out*mask, axis=1)
        out = tf.layers.dense(out, size, name='output', reuse=reuse)

    return out


def attention(Q, K, dim, num_heads, name='attention', reuse=False, mask=None):
    with tf.variable_scope(name, reuse=reuse):
        q = tf.layers.dense(Q, dim, name='query', reuse=reuse)
        k = tf.layers.dense(K, dim, name='key', reuse=reuse)
        v = tf.layers.dense(K, dim, name='value', reuse=reuse)

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
        o = o + tf.layers.dense(o, dim, activation=tf.nn.relu, name='output', reuse=reuse)

    return o

    
def set_transformer(inputs, layers_sizes, num_heads=4, name="set_transformer", reuse=False, mask=None):
    print("Building set transformer {} | sizes: {}".format(name, [inputs.shape[-1]] + layers_sizes))
    
    with tf.variable_scope(name, reuse=reuse):
        out = inputs
        for i, size in enumerate(layers_sizes):
            print(f"Layer: self_attn_{i}: {size}")
            out = attention(out, out, size, num_heads, name=f'self_attn_{i}', reuse=reuse, mask=mask)

        print(f"Layer: pool_attn: {size}")
        seed = tf.get_variable('pool_seed', shape=[1,1,size], dtype=tf.float32, trainable=True, 
                                initializer=tf.contrib.layers.xavier_initializer())
        seed = tf.tile(seed, [tf.shape(out)[0],1,1])
        out = attention(seed, out, size, num_heads, name='pool_attn', reuse=reuse, mask=mask)
        out = tf.squeeze(out, axis=1)
        out = tf.layers.dense(out, size, name='output', reuse=reuse)

    return out

def induced_set_transformer(inputs, layers_sizes, num_heads=4, num_inds=16, name="set_transformer", reuse=False, mask=None):
    print("Building set transformer {} | sizes: {}".format(name, [inputs.shape[-1]] + layers_sizes))

    with tf.variable_scope(name, reuse=reuse):
        out = inputs
        for i, size in enumerate(layers_sizes):
            print(f"Layer: self_attn_{i}: {size}")
            inds = tf.get_variable(f'inds_{i}', shape=[1,num_inds,size], dtype=tf.float32, trainable=True,
                                    initializer=tf.contrib.layers.xavier_initializer())
            inds = tf.tile(inds, [tf.shape(out)[0],1,1])
            tmp = attention(inds, out, size, num_heads, name=f'self_attn_{i}_pre', reuse=reuse, mask=mask)
            out = attention(out, tmp, size, num_heads, name=f'self_attn_{i}_post', reuse=reuse, mask=None)

        print(f"Layer: pool_attn: {size}")
        seed = tf.get_variable('pool_seed', shape=[1,1,size], dtype=tf.float32, trainable=True, 
                                initializer=tf.contrib.layers.xavier_initializer())
        seed = tf.tile(seed, [tf.shape(out)[0],1,1])
        out = attention(seed, out, size, num_heads, name='pool_attn', reuse=reuse, mask=mask)
        out = tf.squeeze(out, axis=1)
        out = tf.layers.dense(out, size, name='output', reuse=reuse)

    return out


def dense_nn(inputs, layers_sizes, name="mlp", reuse=False, output_fn=None,
             dropout_keep_prob=None, batch_norm=False, training=True):
    print("Building mlp {} | sizes: {}".format(name, [inputs.shape[-1]] + layers_sizes))

    with tf.variable_scope(name, reuse=reuse):
        out = inputs
        for i, size in enumerate(layers_sizes):
            print("Layer:", name + '_l' + str(i), size)
            if i > 0 and dropout_keep_prob is not None and training:
                # No dropout on the input layer.
                out = tf.nn.dropout(out, dropout_keep_prob)

            out = tf.layers.dense(
                out,
                size,
                # Add relu activation only for internal layers.
                activation=tf.nn.relu if i < len(layers_sizes) - 1 else None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name=name + '_l' + str(i),
                reuse=reuse
            )

            if batch_norm:
                out = tf.layers.batch_normalization(out, training=training)

        if output_fn:
            out = output_fn(out)

    return out


def gcn_layer(X, A, size, name='gcn_layer', reuse=False):
    '''
    Args:
        X: [B, N, cin]
        A: [B, N, N]
    Return:
        out: [B, N, size]
    '''
    with tf.variable_scope(name, reuse=reuse):
        # AX: outgoing
        D = tf.reduce_sum(A, axis=-1)
        D = tf.where(tf.equal(D, 0), tf.zeros_like(D), tf.pow(D, -0.5))
        Dm = tf.matrix_diag(D)
        Am = tf.matmul(tf.matmul(Dm, A), Dm)
        AX = tf.matmul(Am, X)
        # AtX: incoming
        At = tf.transpose(A, perm=[0,2,1])
        Dt = tf.reduce_sum(At, axis=-1)
        Dt = tf.where(tf.equal(Dt, 0), tf.zeros_like(Dt), tf.pow(Dt, -0.5))
        Dtm = tf.matrix_diag(Dt)
        Atm = tf.matmul(tf.matmul(Dtm, At), Dtm)
        AtX = tf.matmul(Atm, X)
        # residual connection
        inputs = tf.concat([X, AX, AtX], axis=-1)
        res = tf.concat([inputs, -inputs], axis=-1)
        res = tf.nn.relu(res)
        res = tf.layers.dense(res, size,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name=name+'_res',
                reuse=reuse)
        inp = tf.layers.dense(X, size,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name=name+'_inp',
                reuse=reuse)
        out = tf.nn.relu(inp+res)

    return out
        

def gcn(inputs, A, layers_sizes, name='gcn', reuse=False):
    print("Building gcn {} | sizes: {}".format(name, [inputs.shape[-1]] + layers_sizes))

    with tf.variable_scope(name, reuse=reuse):
        out = inputs
        for i, size in enumerate(layers_sizes):
            print("Layer:", name + '_l' + str(i), size)
            out = gcn_layer(out, A, size, name=name+'_l'+str(i), reuse=reuse)

    return out
    