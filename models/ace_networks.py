import tensorflow as tf2
import tensorflow_probability as tfp
from tensorflow.keras import layers as tfl


def proposal_network(
    num_features: int,
    context_units: int = 64,
    mixture_components: int = 10,
    residual_blocks: int = 4,
    hidden_units: int = 512,
    activation: str = "relu",
    dropout: float = 0.0,
    **kwargs
):
    x_o = tfl.Input((num_features,), name="x_o")
    observed_mask = tfl.Input((num_features,), name="observed_mask")

    h = tfl.Concatenate()([x_o, observed_mask])
    h = tfl.Dense(hidden_units)(h)

    for _ in range(residual_blocks):
        res = tfl.Activation(activation)(h)
        res = tfl.Dense(hidden_units)(res)
        res = tfl.Activation(activation)(res)
        res = tfl.Dropout(dropout)(res)
        res = tfl.Dense(hidden_units)(res)
        h = tfl.Add()([h, res])

    h = tfl.Activation(activation)(h)
    h = tfl.Dense(num_features * (3 * mixture_components + context_units))(h)
    h = tfl.Reshape([num_features, 3 * mixture_components + context_units])(h)

    context = h[..., :context_units]
    params = h[..., context_units:]

    def create_proposal_dist(t):
        logits = t[..., :mixture_components]
        means = t[..., mixture_components:-mixture_components]
        scales = tf2.nn.softplus(t[..., -mixture_components:]) + 1e-3
        components_dist = tfp.distributions.Normal(
            loc=tf2.cast(means, tf2.float32), scale=tf2.cast(scales, tf2.float32)
        )
        return tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                logits=tf2.cast(logits, tf2.float32)
            ),
            components_distribution=components_dist,
        )

    proposal_dist = tfp.layers.DistributionLambda(create_proposal_dist)(params)

    return tf2.keras.Model([x_o, observed_mask], [proposal_dist, context], **kwargs)


def energy_network(
    num_features: int,
    context_units: int,
    residual_blocks: int = 4,
    hidden_units: int = 128,
    activation: str = "relu",
    dropout: float = 0.0,
    energy_clip: float = 30.0,
    **kwargs
):
    x_u_i = tfl.Input((), name="x_u_i")
    u_i = tfl.Input((), name="u_i", dtype=tf2.int32)
    context = tfl.Input((context_units,), name="context")

    u_i_one_hot = tf2.one_hot(u_i, num_features)

    h = tfl.Concatenate()([tf2.expand_dims(x_u_i, axis=-1), u_i_one_hot, context])
    h = tfl.Dense(hidden_units)(h)

    for _ in range(residual_blocks):
        res = tfl.Activation(activation)(h)
        res = tfl.Dense(hidden_units)(res)
        res = tfl.Activation(activation)(res)
        res = tfl.Dropout(dropout)(res)
        res = tfl.Dense(hidden_units)(res)
        h = tfl.Add()([h, res])

    h = tfl.Activation(activation)(h)
    h = tfl.Dense(1)(h)

    energies = tf2.nn.softplus(h)
    energies = tf2.clip_by_value(energies, 0.0, energy_clip)
    negative_energies = -energies

    return tf2.keras.Model([x_u_i, u_i, context], negative_energies, **kwargs)
