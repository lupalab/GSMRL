import os
import logging
import numpy as np
import tensorflow as tf
from collections import namedtuple, defaultdict
from scipy.special import softmax
from pprint import pformat
import random

from utils.nn_utils import dense_nn, set_transformer, induced_set_transformer
from utils.memory import ReplayMemory
from utils.visualize import plot_dict

logger = logging.getLogger(__name__)

class PPOPolicy(object):
    def __init__(self, hps, env, split):
        self.hps = hps
        self.env = env
        self.act_size = self.hps.act_size

        g = tf.Graph()
        with g.as_default():
            # open a session
            config = tf.ConfigProto()
            config.log_device_placement = True
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=g)
            # build model
            self._build_networks()
            self._build_train_ops()
            # initialize
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            self.writer = tf.summary.FileWriter(self.hps.exp_dir + '/summary')

    def save(self, filename='params'):
        fname = f'{self.hps.exp_dir}/weights/{filename}.ckpt'
        self.saver.save(self.sess, fname)
        if self.hps.finetune_env:
            fname = f'{self.hps.exp_dir}/weights/env_{filename}.ckpt'
            self.env.saver.save(self.env.sess, fname)

    def load(self, filename='params'):
        fname = f'{self.hps.exp_dir}/weights/{filename}.ckpt'
        self.saver.restore(self.sess, fname)
        if self.hps.finetune_env:
            fname = f'{self.hps.exp_dir}/weights/env_{filename}.ckpt'
            self.env.saver.restore(self.env.sess, fname)

    def act(self, state, mask, future, hard=False):
        '''
        state: [B,d] observed dimensions with values
        mask: [B,d] binary mask indicating observed dimensions
              1: observed   0: unobserved
        future: [B,d*n]
        action: [B] sample an action to take
        prediction: [B,K] prediction from partial observation
        '''
        probas, prediction = self.sess.run([self.actor_proba, self.predictor],
                                        feed_dict={self.state: state,
                                                   self.mask: mask,
                                                   self.future: future})
        
        # logger.info(f'probas:  {probas}')
        # logger.info(f'self.x:  {self.x}')
        # for i, vals in enumerate(self.x):
        #     for j, val in enumerate(vals):
        #         if val == 0:
        #             probas[i][j] = 0
        # sum = []
        # for i in probas:
        #     sum.append(np.sum(i))

        # for i, vals in enumerate(probas):
        #     for j, val in enumerate(vals):
        #         probas[i][j] = probas[i][j] / sum[i]    
        if hard:
            action = np.array([np.argmax(p) for p in probas])
        else:
            action = np.array([np.random.choice(self.act_size, p=p) for p in probas])

        return action, prediction

    def scope_vars(self, scope, only_trainable=True):
        collection = tf.GraphKeys.TRAINABLE_VARIABLES if only_trainable else tf.GraphKeys.VARIABLES
        variables = tf.get_collection(collection, scope=scope)
        assert len(variables) > 0
        logger.info(f"Variables in scope '{scope}':")
        for v in variables:
            logger.info("\t" + str(v))
        return variables

    def _build_networks(self):
        d = self.hps.dimension
        self.state = tf.placeholder(tf.float32, shape=[None, d], name='state')
        self.mask = tf.placeholder(tf.float32, shape=[None, d], name='mask')
        self.future = tf.placeholder(tf.float32, shape=[None, d*self.env.n_future], name='future')
        self.action = tf.placeholder(tf.int32, shape=[None], name='action')
        self.next_state = tf.placeholder(tf.float32, shape=[None, d], name='next_state')
        self.reward = tf.placeholder(tf.float32, shape=[None], name='reward')
        self.done = tf.placeholder(tf.float32, shape=[None], name='done_flag')

        self.old_logp_a = tf.placeholder(tf.float32, shape=[None], name='old_logp_a')
        if self.env.task == 'reg':
            self.p_target = tf.placeholder(tf.float32, shape=[None,self.hps.n_target], name='p_target')
        else:
            self.p_target = tf.placeholder(tf.float32, shape=[None], name='p_target')
        self.v_target = tf.placeholder(tf.float32, shape=[None], name='v_target')
        self.adv = tf.placeholder(tf.float32, shape=[None], name='return')

        with tf.variable_scope('embedding'):
            if self.hps.embed_type == 'set':
                state = tf.expand_dims(self.state, axis=-1)
                mask = tf.expand_dims(self.mask, axis=-1)
                future = tf.transpose(tf.reshape(self.future, [-1,self.env.n_future,d]), perm=[0,2,1])
                index = tf.tile(tf.expand_dims(tf.eye(d), axis=0), [tf.shape(state)[0],1,1])
                inputs = tf.concat([state, future, mask, index], axis=-1)
                if self.hps.num_induced_points > 0:
                    embed = induced_set_transformer(inputs, self.hps.embed_layers, self.hps.num_heads, num_inds=self.hps.num_induced_points)
                else:
                    embed = set_transformer(inputs, self.hps.embed_layers, self.hps.num_heads)
                self.embed_vars = self.scope_vars('embedding')
            else:
                embed = tf.concat([self.state, self.future, self.mask], axis=-1)
                self.embed_vars = []

        with tf.variable_scope('actor'):
            # Actor: action probabilities
            actor_layers = self.hps.actor_layers + [self.act_size]
            self.actor = dense_nn(embed, actor_layers, name='actor')
            if self.env.task == 'ts':
                assert d % self.hps.time_steps == 0
                assert self.act_size == self.hps.time_steps + 1
                logits_mask = tf.reshape(self.mask, [tf.shape(self.mask)[0], self.hps.time_steps, -1])
                logits_mask = logits_mask[:,:,0]
                cum_mask = tf.cumsum(logits_mask, axis=1, reverse=True)
                logits_mask = tf.where(tf.equal(cum_mask, 0.), tf.zeros_like(logits_mask), tf.ones_like(logits_mask))
                logits_mask = tf.concat([logits_mask, tf.zeros([tf.shape(self.mask)[0], 1])], axis=1)
            elif hasattr(self.env, 'terminal_act'):
                assert self.act_size == d + 1
                logits_mask = tf.concat([self.mask, tf.zeros([tf.shape(self.mask)[0], 1])], axis=1)
            else:
                assert self.act_size == d
                logits_mask = self.mask
            inf_tensor = -tf.ones_like(self.actor) * np.inf
            self.actor_logits = tf.where(tf.equal(logits_mask, 0), self.actor, inf_tensor)
            self.actor_proba = tf.nn.softmax(self.actor_logits)
            self.actor_log_proba = tf.nn.log_softmax(self.actor_logits)
            self.actor_entropy = tf.distributions.Categorical(probs=self.actor_proba).entropy()
            index = tf.stack([tf.range(tf.shape(self.action)[0]), self.action], axis=1)
            self.logp_a = tf.gather_nd(self.actor_log_proba, index)
            self.actor_vars = self.scope_vars('actor')

        with tf.variable_scope('predictor'):
            # Predictor: predict target variable
            predictor_layers = self.hps.predictor_layers + [self.hps.n_target]
            self.predictor = dense_nn(embed, predictor_layers, name='predictor')
            self.predictor_vars = self.scope_vars('predictor')

        with tf.variable_scope('critic'):
            # Critic: action value (V value)
            critic_layers = self.hps.critic_layers + [1]
            self.critic = tf.squeeze(dense_nn(embed, critic_layers, name='critic'))
            self.critic_vars = self.scope_vars('critic')

    def _build_train_ops(self):
        self.lr_a = tf.placeholder(tf.float32, shape=None, name='learning_rate_actor')
        self.lr_p = tf.placeholder(tf.float32, shape=None, name='learning_rate_predictor')
        self.lr_c = tf.placeholder(tf.float32, shape=None, name='learning_rate_critic')
        self.clip_range = tf.placeholder(tf.float32, shape=None, name='ratio_clip_range')

        with tf.variable_scope('actor_train'):
            ratio = tf.exp(self.logp_a - self.old_logp_a)
            ratio_clipped = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            loss_a = - tf.reduce_mean(tf.minimum(self.adv * ratio, self.adv * ratio_clipped))
            if self.hps.ent_coef > 0:
                loss_a -= tf.reduce_mean(self.actor_entropy) * self.hps.ent_coef

            optim_a = tf.train.AdamOptimizer(self.lr_a)
            grads_and_vars = optim_a.compute_gradients(loss_a, var_list=self.actor_vars+self.embed_vars)
            grads_a, vars_a = zip(*grads_and_vars)
            if self.hps.clip_grad_norm > 0:
                grads_a, gnorm_a = tf.clip_by_global_norm(grads_a, clip_norm=self.hps.clip_grad_norm)
                gnorm_a = tf.check_numerics(gnorm_a, "Gradient norm is NaN or Inf.")
                tf.summary.scalar('gnorm_a', gnorm_a)
            grads_and_vars = zip(grads_a, vars_a)
            self.train_op_a = optim_a.apply_gradients(grads_and_vars)

        with tf.variable_scope('predictor_train'):
            if self.env.task == 'reg':
                loss_p = tf.reduce_sum(tf.square(self.p_target-self.predictor), axis=1)
            else:
                p_target = tf.cast(self.p_target, tf.int64)
                loss_p = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predictor, labels=p_target)
            loss_p = tf.reduce_mean(loss_p * self.done)

            optim_p = tf.train.AdamOptimizer(self.lr_p)
            grads_and_vars = optim_p.compute_gradients(loss_p, var_list=self.predictor_vars+self.embed_vars)
            grads_p, vars_p = zip(*grads_and_vars)
            if self.hps.clip_grad_norm > 0:
                grads_p, gnorm_p = tf.clip_by_global_norm(grads_p, clip_norm=self.hps.clip_grad_norm)
                gnorm_p = tf.check_numerics(gnorm_p, "Gradient norm is NaN or Inf.")
                tf.summary.scalar('gnorm_p', gnorm_p)
            grads_and_vars = zip(grads_p, vars_p)
            self.train_op_p = optim_p.apply_gradients(grads_and_vars)

        with tf.variable_scope('critic_train'):
            loss_c = tf.reduce_mean(tf.square(self.v_target - self.critic))

            optim_c = tf.train.AdamOptimizer(self.lr_c)
            grads_and_vars = optim_c.compute_gradients(loss_c, var_list=self.critic_vars+self.embed_vars)
            grads_c, vars_c = zip(*grads_and_vars)
            if self.hps.clip_grad_norm > 0:
                grads_c, gnorm_c = tf.clip_by_global_norm(grads_c, clip_norm=self.hps.clip_grad_norm)
                gnorm_c = tf.check_numerics(gnorm_c, "Gradient norm is NaN or Inf.")
                tf.summary.scalar('gnorm_c', gnorm_c)
            grads_and_vars = zip(grads_c, vars_c)
            self.train_op_c = optim_c.apply_gradients(grads_and_vars)

        self.train_ops = tf.group(self.train_op_a, self.train_op_p, self.train_op_c)

        with tf.variable_scope('summary'):
            self.ep_reward = tf.placeholder(tf.float32, name='episode_reward')

            self.summary = [
                tf.summary.scalar('loss/adv', tf.reduce_mean(self.adv)),
                tf.summary.scalar('loss/ratio', tf.reduce_mean(ratio)),
                tf.summary.scalar('loss/loss_actor', loss_a),
                tf.summary.scalar('loss/loss_predictor', loss_p),
                tf.summary.scalar('loss/loss_critic', loss_c),
                tf.summary.scalar('episode_reward', self.ep_reward)
            ]

            self.summary += [tf.summary.histogram('vars/' + v.name, v)
                            for v in vars_a if v is not None]
            self.summary += [tf.summary.histogram('vars/' + v.name, v)
                            for v in vars_p if v is not None]
            self.summary += [tf.summary.histogram('vars/' + v.name, v)
                            for v in vars_c if v is not None]

            self.summary += [tf.summary.scalar('grads/' + g.name, tf.norm(g))
                            for g in grads_a if g is not None]
            self.summary += [tf.summary.scalar('grads/' + g.name, tf.norm(g))
                            for g in grads_p if g is not None]
            self.summary += [tf.summary.scalar('grads/' + g.name, tf.norm(g))
                            for g in grads_c if g is not None]

            self.merged_summary = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

    def _generate_rollout(self, buffer):
        s, m = self.env.reset() # [B,d]
        obs = []
        masks = []
        futures = []
        actions = []
        rewards = []
        flags = []
        episode_reward = np.zeros([s.shape[0]], dtype=np.float32)

        logger.info('start rollout.')
        done = np.zeros([s.shape[0]], dtype=np.bool)
        while not np.all(done):
            logger.debug(f'mask: {m}')
            f = self.env.peek(s, m)
            a_orig, p = self.act(s, m, f) # [B]
            logger.debug(f'action: {a_orig}')
            a = a_orig.copy()
            a[done] = -1 # empty action
            s_next, m_next, r, done = self.env.step(a, p)
            logger.debug(f'done: {done}')
            obs.append(s)
            masks.append(m)
            futures.append(f)
            actions.append(a_orig)
            rewards.append(r)
            flags.append(done)
            episode_reward += r
            s, m = s_next, m_next
        logger.info('rollout finished.')

        logger.debug(f'mask:\n{np.concatenate(masks)}')
        logger.debug(f'state:\n{np.concatenate(obs)}')
        logger.debug(f'future:\n{np.concatenate(futures)}')
        logger.debug(f'action:\n{np.concatenate(actions)}')
        logger.debug(f'flags:\n{np.concatenate(flags)}')
        logger.debug(f'reward:\n{np.concatenate(rewards)}')
        
        # length of the episode.
        T = len(rewards)
        
        # compute the current log pi(a|s) and predicted v values.
        with self.sess.as_default():
            if False:
                logp_a = self.logp_a.eval({self.action: np.concatenate(actions), 
                                        self.state: np.concatenate(obs),
                                        self.mask: np.concatenate(masks),
                                        self.future: np.concatenate(futures)})
                logp_a = logp_a.reshape([T, -1])
                logger.debug(f'logp_a:\n{logp_a}')
                assert not np.any(np.isnan(logp_a)), 'logp_a contains NaN values.'
                assert not np.any(np.isinf(logp_a)), 'logp_a contains Inf values.'

                v_pred = self.critic.eval({self.state: np.concatenate(obs),
                                        self.mask: np.concatenate(masks),
                                        self.future: np.concatenate(futures)})
                v_pred = v_pred.reshape([T, -1])
                logger.debug(f'v_pred:\n{v_pred}')
                assert not np.any(np.isnan(v_pred)), 'v_pred contains NaN values.'
                assert not np.any(np.isinf(v_pred)), 'v_pred contains Inf values.'
            else:
                logp_a_list = []
                v_pred_list = []
                for at, xt, mt, ft in zip(actions, obs, masks, futures):
                    logp_a = self.logp_a.eval({self.action: at,
                                               self.state: xt,
                                               self.mask: mt,
                                               self.future: ft})
                    logp_a_list.append(logp_a)
                    logger.debug(f'logp_a:\n{logp_a}')
                    assert not np.any(np.isnan(logp_a)), 'logp_a contains NaN values.'
                    assert not np.any(np.isinf(logp_a)), 'logp_a contains Inf values.'

                    v_pred = self.critic.eval({self.state: xt,
                                               self.mask: mt,
                                               self.future: ft})
                    v_pred_list.append(v_pred)
                    logger.debug(f'v_pred:\n{v_pred}')
                    assert not np.any(np.isnan(v_pred)), 'v_pred contains NaN values.'
                    assert not np.any(np.isinf(v_pred)), 'v_pred contains Inf values.'
                logp_a = np.stack(logp_a_list)
                v_pred = np.stack(v_pred_list)

        # record this batch
        logger.info('record this batch.')
        x = self.env.x.copy()
        y = self.env.y.copy()
        n_rec = 0
        for i in range(s.shape[0]):
            done = [f[i] for f in flags]
            max_T = np.min(np.where(done)[0])
            n_rec += max_T
            state = [s[i] for s in obs][:max_T+1]
            mask = [m[i] for m in masks][:max_T+1]
            future = [f[i] for f in futures][:max_T+1]
            action = [a[i] for a in actions][:max_T+1]
            reward = [r[i] for r in rewards][:max_T+1]
            logp = logp_a[:max_T+1, i]
            vp = v_pred[:max_T+1, i]
            next_state = s_next[i]
            next_mask = m_next[i]

            # Compute TD errors
            td_errors = [reward[t] + self.hps.gamma * vp[t + 1] - vp[t] for t in range(max_T)]
            td_errors += [reward[max_T] + self.hps.gamma * 0.0 - vp[max_T]]  # handle the terminal state.

            # Estimate advantage backwards.
            advs = []
            adv_so_far = 0.0
            for delta in td_errors[::-1]:
                adv_so_far = delta + self.hps.gamma * self.hps.lam * adv_so_far
                advs.append(adv_so_far)
            advs = advs[::-1]
            assert len(advs) == max_T+1

            # Estimate critic target
            vt = np.array(advs) + np.array(vp)

            # add into the memory buffer
            for t, (s, m, f, a, sn, mn, r, old_logp_a, v_target, adv) in enumerate(zip(
                state, mask, future, action, 
                np.array(state[1:] + [next_state]), np.array(mask[1:] + [next_mask]),
                reward, logp, vt, advs)):
                done = float(t == max_T)
                buffer.add(buffer.tuple_class(x[i], y[i], s, m, f, a, sn, mn, r, done, old_logp_a, v_target, adv))
        logger.info(f'record done: {n_rec} transitions added.')

        return np.mean(episode_reward), n_rec

    def _ratio_clip_fn(self, n_iter):
        clip = self.hps.ratio_clip_range
        if self.hps.ratio_clip_decay:
            delta = clip / self.hps.train_iters
            clip -= delta * n_iter

        return max(0.0, clip)

    def train(self):
        BufferRecord = namedtuple('BufferRecord', ['x', 'y', 's', 'm', 'f', 'a', 's_next', 'm_next', 'r', 'done', 
                                                   'old_logp_a', 'v_target', 'adv'])
        buffer = ReplayMemory(tuple_class=BufferRecord, capacity=self.hps.buffer_size)
        
        # pretrain
        buffer.clean()
        for n_iter in range(self.hps.pretrain_iters):
            _ = self._generate_rollout(buffer)

            for batch in buffer.loop(self.hps.batch_size, self.hps.epochs):
                _ = self.sess.run(self.train_op_p,
                        feed_dict={self.lr_p: self.hps.lr_p,
                                   self.state: batch['s'],
                                   self.mask: batch['m'],
                                   self.future: batch['f'],
                                   self.done: batch['done'],
                                   self.p_target: batch['y']
                                   })

        # train
        reward_history = []
        reward_averaged = []
        best_reward = -np.inf
        step = 0
        total_rec = 0

        buffer.clean()
        for n_iter in range(self.hps.train_iters):
            clip = self._ratio_clip_fn(n_iter)
            if self.hps.clean_buffer:
                buffer.clean()
            ep_reward, n_rec = self._generate_rollout(buffer)
            reward_history.append(ep_reward)
            reward_averaged.append(np.mean(reward_history[-10:]))
            total_rec += n_rec

            for batch in buffer.loop(self.hps.batch_size, self.hps.epochs):
                _, summ_str = self.sess.run(
                 [self.train_ops, self.merged_summary],
                 feed_dict={self.lr_a: self.hps.lr_a,
                            self.lr_p: self.hps.lr_p,
                            self.lr_c: self.hps.lr_c,
                            self.clip_range: clip,
                            self.state: batch['s'],
                            self.mask: batch['m'],
                            self.future: batch['f'],
                            self.action: batch['a'],
                            self.next_state: batch['s_next'],
                            self.reward: batch['r'],
                            self.done: batch['done'],
                            self.old_logp_a: batch['old_logp_a'],
                            self.p_target: batch['y'],
                            self.v_target: batch['v_target'],
                            self.adv: batch['adv'],
                            self.ep_reward: np.mean(reward_history[-10:]) if reward_history else 0.0,
                            })
                self.writer.add_summary(summ_str, step)
                step += 1

            if self.hps.finetune_env == 1:
                for batch in buffer.loop(self.hps.finetune_batch_size, self.hps.finetune_epochs):
                    self.env.finetune(batch)
            if self.hps.finetune_env == 2:
                for _ in range(self.hps.finetune_iters):
                    batch = buffer.sample(self.hps.finetune_batch_size)
                    self.env.finetune(batch)

            if self.hps.log_freq > 0 and (n_iter+1) % self.hps.log_freq == 0:
                logger.info("[iteration:{}/step:{}], best:{}, avg:{:.2f}, clip:{:.2f}; {} transitions.".format(
                    n_iter, step, np.max(reward_history), np.mean(reward_history[-10:]), clip, total_rec
                ))

            if self.hps.eval_freq > 0 and (n_iter+1) % self.hps.eval_freq == 0:
                self.evaluate(load=False)

            if self.hps.save_freq > 0 and (n_iter+1) % self.hps.save_freq == 0:
                self.save()

            if np.mean(reward_history[-10:]) > best_reward:
                best_reward = np.mean(reward_history[-10:])
                self.save('best')

        # FINISH
        self.save()
        logger.info("[FINAL] episodes: {}, Max reward: {}, Average reward: {}".format(
            len(reward_history), np.max(reward_history), np.mean(reward_history)))
        data_dict = {
            'reward': reward_history,
            'reward_smooth10': reward_averaged,
        }
        plot_dict(f'{self.hps.exp_dir}/learning_curve.png', data_dict, xlabel='episode')

    def evaluate(self, load=True, hard=False, max_batches=10):
        if load: self.load('best')
        metrics = defaultdict(list)
        transitions = []
        init = True
        num_batches = 0
        while True: # iterate over dataset
            num_batches += 1
            s, m = self.env.reset(loop=False, init=init)
            init = False
            if s is None or m is None:
                break # stop iteration
            if num_batches > max_batches:
                break
            num_acquisition = np.zeros([s.shape[0]], dtype=np.float32)
            episode_reward = np.zeros([s.shape[0]], dtype=np.float32)
            transition = np.zeros_like(m)
            done = np.zeros([s.shape[0]], dtype=np.bool)
            while not np.all(done):
                f = self.env.peek(s, m)
                a, p = self.act(s, m, f, hard=hard)
                a[done] = -1
                s, m, r, done = self.env.step(a, p)
                episode_reward += r
                num_acquisition += ~done
                transition += m
            metrics['episode_reward'].append(episode_reward)
            metrics['num_acquisition'].append(num_acquisition)
            transitions.append(transition.astype(np.int32))
            # evaluate the final state
            eval_dict = self.env.evaluate(s, m, p)
            for k, v in eval_dict.items():
                metrics[k].append(v)

        # concat metrics
        average_metrics = defaultdict(float)
        for k, v in metrics.items():
            metrics[k] = np.concatenate(v)
            average_metrics[k] = np.mean(metrics[k])

        # transitions
        transitions = np.concatenate(transitions)
        action_freq = (transitions != 0).astype(np.float32).sum(axis=0)
        logger.info('action frequency:')
        logger.info(pformat({i:action_freq[i] for i in range(len(action_freq))}))
        logger.info('example transitions:')
        for i in range(5):
            logger.info(transitions[i])

        # log
        logger.info('#'*20)
        logger.info('evaluate:')
        for k, v in average_metrics.items():
            logger.info(f'{k}: {v}')

        return {'metrics': metrics, 'transitions': transitions}