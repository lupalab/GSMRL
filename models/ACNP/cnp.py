import tensorflow as tf

from .modules import *

class CNP(object):
    def __init__(self, hps, scope=None):
        self.hps = hps

        self.deterministic_encoder = DeterministicEncoder(self.hps.deterministic_layers, self.hps.attention)
        self.latent_prior = LatentEncoder(self.hps.latent_layers, self.hps.num_latents, self.hps.num_comps, name='prior')
        self.latent_posterior = LatentEncoder(self.hps.latent_layers, self.hps.num_latents, 1, name='posterior')
        self.decoder = Decoder(self.hps.decoder_layers+[self.hps.y_size*2])
        self.classifier = Classifier(self.hps.classifier_layers+[self.hps.n_classes])
    
    def build(self, xc, yc, mc, xt, yt, mt, lab):
        prior = self.latent_prior(xc, yc, mc)
        posterior = self.latent_posterior(xt, yt, mt, tf.one_hot(lab, self.hps.n_classes))

        deterministic_rep = self.deterministic_encoder(xc, yc, mc, xt)
        post_latent_rep = posterior.sample()
        pri_latent_rep = prior.sample()

        # log_likel
        if self.hps.mix_latent:
            mask = tf.cast(tf.random.uniform([tf.shape(xc)[0],1], 0, 2, dtype=tf.int64), tf.float32)
            latent_rep = post_latent_rep * mask + pri_latent_rep * (1.-mask)
        else:
            latent_rep = post_latent_rep
        logits = self.classifier(latent_rep)
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=lab)
        latent_rep = tf.tile(tf.expand_dims(latent_rep, axis=1), [1,tf.shape(xt)[1],1])
        representation = tf.concat([deterministic_rep, latent_rep], axis=-1)
        dist = self.decoder(representation, xt)
        logpx = tf.reduce_sum(dist.log_prob(yt) * tf.squeeze(mt, axis=2), axis=1)
        num_points = tf.reduce_sum(mt, axis=[1,2]) + 1e-8
        logpx = logpx / num_points
        if self.hps.num_comps == 1:
            kl = tfd.kl_divergence(posterior, prior)
        else:
            post_samples = posterior.sample(50)
            kl = posterior.log_prob(post_samples) - prior.log_prob(post_samples)
            kl = tf.reduce_mean(kl, axis=0)
        self.log_likel = -self.hps.lambda_xent * xent + logpx - self.hps.lambda_kl * kl

        # sample
        logits = self.classifier(pri_latent_rep)
        self.prob = tf.nn.softmax(logits)
        self.pred = tf.argmax(logits, axis=-1)
        self.xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=lab)
        pri_latent_rep = tf.tile(tf.expand_dims(pri_latent_rep, axis=1), [1,tf.shape(xt)[1],1])
        representation = tf.concat([deterministic_rep, pri_latent_rep], axis=-1)
        dist = self.decoder(representation, xt)
        self.pre_yt_loc = dist.loc
        self.pre_yt_scale = dist.scale.diag

        # sample with predicted label
        posterior = self.latent_posterior(xc, yc, mc, tf.one_hot(self.pred, self.hps.n_classes))
        post_latent_rep = posterior.sample()
        post_latent_rep = tf.tile(tf.expand_dims(post_latent_rep, axis=1), [1,tf.shape(xt)[1],1])
        representation = tf.concat([deterministic_rep, post_latent_rep], axis=-1)
        dist = self.decoder(representation, xt)
        self.post_yt_loc = dist.loc
        self.post_yt_scale = dist.scale.diag
