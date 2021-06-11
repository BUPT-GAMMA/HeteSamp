import numpy as np
import tensorflow as tf
from tensorflow.contrib.metrics import f1_score as f1_measure

from tools.layers import dense_layers


class HeteSamp:
    def __init__(self, FLAGS, configs, global_step, is_train="train"):

        self.configs = configs

        self.global_step = global_step
        learning_rate = FLAGS.learning_rate
        learning_algo = FLAGS.learning_algo

        self.sample_method = FLAGS.sample_method
        if self.sample_method == "ni":
            self.alpha = 0
        else:
            self.alpha = FLAGS.alpha
        self.beta = FLAGS.beta
        self.gamma = FLAGS.gamma
        self.xi = FLAGS.xi
        self.keep_prob = FLAGS.keep_prob
        self.variance_reduction = FLAGS.variance_reduction

        # dataset info
        self.n_node_type = configs["num_node_types"]
        self.n_edge_type = configs["num_edge_types"]
        self.num_negs = configs["num_negs"]
        self.edge_feats_lengths = configs["edge_feats_lengths"]
        self.num_nodes = configs["num_nodes"]
        self.nclass = configs["nclass"]
        self.utilize_edge_feats = configs["utilize_edge_feats"]
        # learn edge weight with parameters
        self.task = configs["task"]
        self.sampled_numbers = FLAGS.num_sample * 1.0

        with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
            if learning_algo == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            elif learning_algo == "sgd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            else:
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)

        self.embedding_dim = FLAGS.embedding_dim
        init_range = np.sqrt(3.0 / (self.num_nodes + self.embedding_dim))

        random_initer = tf.random_uniform([self.num_nodes, self.embedding_dim], minval=-init_range, maxval=init_range,
                                          dtype=tf.float32)

        with tf.variable_scope("parameters", reuse=tf.AUTO_REUSE):
            self.embedding_table = tf.get_variable('node_embedding_table',
                                                   initializer=random_initer)
            self.W_nbrs = tf.get_variable("W_nbrs", [self.n_edge_type, self.embedding_dim, self.embedding_dim],
                                          dtype=tf.float32, initializer=tf.glorot_normal_initializer())

            self.feat_parameters = []
            for i in range(self.n_edge_type):
                init_range2 = np.sqrt(3.0 / (self.edge_feats_lengths[i] + 1))

                self.feat_parameters.append(
                    tf.get_variable("edge_feat_paramters_{}".format(i),
                                    initializer=tf.random_uniform([self.edge_feats_lengths[i], 1], minval=-init_range2,
                                                                  maxval=init_range2, dtype=tf.float32)))
            self.node_bias = tf.get_variable("node_bias", [self.n_node_type, 1], dtype=tf.float32,
                                             initializer=tf.zeros_initializer())

            # ep parameters

    def construct_task_outs(self, keep_prob=1.0):
        # sp_info: supervised information
        with tf.variable_scope("task_graph", reuse=tf.AUTO_REUSE):
            if self.task == "MCP":
                self.nids = tf.placeholder(tf.int32, [None, ], "supervised_node_ids")
                self.labels = tf.placeholder(tf.int32, [None, ], "supervised_labels")
                node_embeds = tf.nn.embedding_lookup(self.embedding_table, self.nids)
                logits = dense_layers(node_embeds, self.nclass, "mcp_logits", keep_prob=keep_prob, norm_rate=self.beta,
                                      activation=None)
                labels = tf.one_hot(self.labels, self.nclass, dtype=tf.float32)
                task_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

                f1_value, update_f1 = f1_measure(labels, tf.nn.softmax(logits, axis=1))
                auc_value, update_auc = tf.metrics.auc(labels, tf.nn.softmax(logits, axis=1))
                return [update_f1, update_auc, "f1_score", "auc_score"], tf.nn.softmax(logits, axis=1), task_loss

            elif self.task == "LP":
                self.lids = tf.placeholder(tf.int32, [None, ], "supervised_pair_ids_left_LP")
                self.rids = tf.placeholder(tf.int32, [None, ], "supervised_pair_ids_right_LP")
                self.labels = tf.placeholder(tf.float32, [None, ], "supervised_labels_LP")

                left_embeds = tf.nn.embedding_lookup(self.embedding_table, self.lids)
                right_embeds = tf.nn.embedding_lookup(self.embedding_table, self.rids)

                logits = dense_layers(tf.concat([left_embeds, right_embeds], axis=1), 1, "logits", keep_prob=1.0,
                                      norm_rate=self.beta, activation=None)
                task_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(self.labels, [-1, 1]),
                                                            logits=tf.reshape(logits, [-1, 1])))

                f1_value, update_f1 = f1_measure(self.labels, tf.nn.sigmoid(logits))
                auc_value, update_auc = tf.metrics.auc(self.labels, tf.nn.sigmoid(logits))
                return [update_f1, update_auc, "f1_score", "auc_score"], tf.nn.sigmoid(logits), task_loss

            else:
                return [], tf.Variable([], tf.float32), tf.constant(0.0, dtype=tf.float32)

    def construct_ep_outs(self, keep_prob=1.0, activation_fn=tf.sigmoid):
        with tf.variable_scope("ep_graph", reuse=tf.AUTO_REUSE):
            self.nids_ep = tf.placeholder(tf.int32, [None, ], "nids_ep")
            self.nids_types = tf.placeholder(tf.int32, [None, ], "ep_nids_types")
            self.neg_ids = tf.placeholder(tf.int32, [None, None], "ep_neg_ids")

            self.nbr_feats = []
            self.edge_weight = []
            self.nbr_ids = []
            self.seg_ids = []
            self.edge_types = []
            for i in range(self.n_edge_type):
                self.nbr_feats.append(
                    tf.placeholder(tf.float32, [None, None], "edge_feats_{}".format(i))
                )
                self.seg_ids.append(tf.placeholder(tf.int32, [None, ], "ep_nbr_segs_{}".format(i)))
                self.nbr_ids.append(tf.placeholder(tf.int32, [None, ], "ep_nbr_ids_{}".format(i)))
                self.edge_weight.append(tf.placeholder(tf.float32, [None, ], "edge_weight_{}".format(i)))
                self.edge_types.append(tf.placeholder(tf.int32, [None, ], "edge_types_{}".format(i)))

            self.batch_size = tf.placeholder(tf.int32, name="batch_size")

            if self.sample_method == "general":
                loss_ep = self.calculate_loss_ep(self.batch_size, self.nids_ep, self.nids_types, self.neg_ids,
                                                 self.seg_ids, self.nbr_ids, self.nbr_feats, self.edge_weight,
                                                 self.edge_types, activation_fn=activation_fn, keep_prob=keep_prob)
                loss_variance = 0
            elif self.sample_method == "nil":
                loss_ep = tf.constant(0.0, tf.float32)
                loss_variance = 0
            elif self.sample_method == "fast" or self.sample_method == "random":
                loss_ep = self.calculate_loss_ep(self.batch_size, self.nids_ep, self.nids_types, self.neg_ids,
                                                 self.seg_ids, self.nbr_ids, self.nbr_feats, self.edge_weight,
                                                 self.edge_types, activation_fn=activation_fn, keep_prob=keep_prob)
                loss_variance = 0
            else:
                self.edge_count = [tf.placeholder(tf.int32, [None, ], "edge_count_{}".format(i)) for i in
                                   range(self.n_edge_type)]
                loss_ep, loss_variance = self.calculate_loss_ep_adaptive(self.batch_size, self.nids_ep, self.nids_types,
                                                                         self.neg_ids,
                                                                         self.seg_ids, self.nbr_ids, self.nbr_feats,
                                                                         self.edge_weight,
                                                                         self.edge_types, self.edge_count,
                                                                         activation_fn=activation_fn,
                                                                         keep_prob=keep_prob)
                # loss_variance = tf.Print(loss_variance, [loss_variance])

        return loss_ep, loss_variance

    def calculate_loss_ep(self, batch_size, ids, id_types, negs, nbr_segs, nbr_ids, nbr_feats, edge_weight, edge_types,
                          activation_fn=tf.sigmoid, keep_prob=1.0):

        embs = tf.nn.embedding_lookup(self.embedding_table, ids)
        negs_lookup = tf.nn.embedding_lookup(self.embedding_table, negs)

        type_embeds_all = []
        type_embeds_segs_all = []

        for i in range(self.n_edge_type):
            type_embed_list = tf.cond(
                tf.size(nbr_ids[i]) > 0,
                false_fn=lambda: [tf.constant([[0.0] * self.embedding_dim], dtype=tf.float32),
                                  tf.reshape(batch_size, (-1,))],
                true_fn=lambda: list(
                    self.calculate_type_embeds(i, nbr_ids[i], nbr_segs[i], edge_weight[i], nbr_feats[i],
                                               tf.nn.sigmoid))

            )
            type_embeds_all.append(type_embed_list[0])
            type_embeds_segs_all.append(type_embed_list[1])

        h_v_base = tf.unsorted_segment_sum(tf.concat(type_embeds_all, axis=0),
                                           tf.reshape(tf.concat(type_embeds_segs_all, axis=0), (-1,)), batch_size + 1)

        h_v_base = h_v_base[:batch_size, :]

        h_v = activation_fn(h_v_base + tf.gather(self.node_bias, id_types))

        pi_neg = self.batch_distance_neg(h_v, negs_lookup)
        pi_pos = self.batch_distance(h_v, embs)
        ep_loss = tf.reduce_mean(self.batch_distance_pair(pi_pos, pi_neg))
        return ep_loss

    def calculate_nbr_feat_weights(self, nbr_feats, edge_type, activation=None, keep_prob=1.0):

        if 0 < keep_prob < 1.0:
            nbr_feats = tf.nn.dropout(nbr_feats, 1 - keep_prob)
        nbr_feat_weight = tf.reshape(activation(tf.matmul(nbr_feats, self.feat_parameters[edge_type])), (-1,))
        return nbr_feat_weight

    def calculate_type_embeds_v0(self, edge_type, nbr_ids, nbr_segs, link_weight, feats, activation):
        nbr_embs = tf.matmul(tf.gather(self.embedding_table, nbr_ids), self.W_nbrs[edge_type])

        nbr_feat_weight = tf.reshape(activation(tf.matmul(feats, self.feat_parameters[edge_type])), (-1,))
        weights = tf.reshape(nbr_feat_weight * link_weight, [-1, 1])
        weight_embeds = tf.multiply(nbr_embs, weights)

        unqiue_nbr_segs = tf.unique(nbr_segs)
        batch_size = tf.size(unqiue_nbr_segs.y)
        new_embeds = tf.unsorted_segment_sum(weight_embeds, unqiue_nbr_segs.idx, batch_size)
        return new_embeds, unqiue_nbr_segs.y

    def calculate_type_embeds(self, edge_type, nbr_ids, nbr_segs, link_weight, feats, activation):
        nbr_embs = tf.gather(self.embedding_table, nbr_ids)

        nbr_feat_weight = tf.reshape(activation(tf.matmul(feats, self.feat_parameters[edge_type])), (-1,))
        weights = tf.reshape(nbr_feat_weight * link_weight, [-1, 1])
        weight_embeds = tf.multiply(nbr_embs, weights)

        unqiue_nbr_segs = tf.unique(nbr_segs)
        batch_size = tf.size(unqiue_nbr_segs.y)
        new_embeds = tf.matmul(tf.unsorted_segment_sum(weight_embeds, unqiue_nbr_segs.idx, batch_size),
                               self.W_nbrs[edge_type])
        return new_embeds, unqiue_nbr_segs.y

    def calculate_loss_ep_adaptive(self, batch_size, ids, id_types, negs, nbr_segs, nbr_ids, nbr_feats, edge_weight,
                                   edge_types, edge_count, activation_fn=tf.sigmoid, keep_prob=1.0):

        embs = tf.nn.embedding_lookup(self.embedding_table, ids)
        negs_lookup = tf.nn.embedding_lookup(self.embedding_table, negs)

        type_embeds_all = []
        type_embeds_segs_all = []

        for i in range(self.n_edge_type):
            type_embed_list = tf.cond(
                tf.size(nbr_ids[i]) > 0,
                false_fn=lambda: [tf.constant([[0.0] * self.embedding_dim], dtype=tf.float32),
                                  tf.reshape(batch_size, (-1,))],
                true_fn=lambda: list(
                    self.calculate_type_embeds(i, nbr_ids[i], nbr_segs[i],
                                               edge_weight[i] * tf.cast(edge_count[i],
                                                                        tf.float32) / self.sampled_numbers,
                                               nbr_feats[i],
                                               tf.nn.sigmoid))
            )
            type_embeds_all.append(type_embed_list[0])
            type_embeds_segs_all.append(type_embed_list[1])

        h_v_base = tf.unsorted_segment_sum(tf.concat(type_embeds_all, axis=0),
                                           tf.reshape(tf.concat(type_embeds_segs_all, axis=0), (-1,)), batch_size + 1)

        h_v_base = h_v_base[:batch_size, :]

        h_v = activation_fn(h_v_base + tf.gather(self.node_bias, id_types))

        pi_neg = self.batch_distance_neg(h_v, negs_lookup)
        pi_pos = self.batch_distance(h_v, embs)
        ep_loss = tf.reduce_mean(self.batch_distance_pair(pi_pos, pi_neg))

        mean_parent_info = tf.reshape(tf.reduce_mean(h_v_base, axis=0), (1, self.embedding_dim))

        type_nbr_embed_list = []
        type_nbr_segs_list = []

        nbr_ids_unique = tf.unique(tf.concat(nbr_ids, axis=0))

        nbr_unique_size = tf.size(nbr_ids_unique.y)
        for i in range(self.n_edge_type):
            type_nbr_embed_i = tf.cond(
                tf.size(nbr_ids[i]) > 0,
                false_fn=lambda: [tf.constant([[0.0] * self.embedding_dim], dtype=tf.float32),
                                  tf.constant([int(1e8)], dtype=tf.int32)],
                true_fn=lambda: list(
                    self.calculate_type_embeds(i, nbr_ids[i], nbr_ids[i],
                                               edge_weight[i] / tf.cast(batch_size, tf.float32),
                                               nbr_feats[i],
                                               tf.nn.sigmoid))
            )

            type_nbr_embed_list.append(type_nbr_embed_i[0])
            type_nbr_segs_list.append(type_nbr_embed_i[1])

        type_nbr_segs_list_unique = tf.unique(tf.concat(type_nbr_segs_list, axis=0))
        print(type_nbr_segs_list)
        h_u_base = tf.unsorted_segment_sum(tf.concat(type_nbr_embed_list, axis=0),
                                           tf.reshape(type_nbr_segs_list_unique.idx, (-1,)),
                                           nbr_unique_size + 1)

        nbr_samples = tf.cast(
            tf.unsorted_segment_mean(tf.concat(edge_count, axis=0), nbr_ids_unique.idx, nbr_unique_size), tf.float32)

        variance_loss = tf.reduce_mean(
            (tf.reduce_mean((mean_parent_info - h_u_base[:-1, :]) ** 2, axis=1) * nbr_samples), axis=0)

        print(mean_parent_info)
        print(type_nbr_segs_list_unique.idx)
        variance_loss = tf.Print(variance_loss, [variance_loss])

        return ep_loss, variance_loss

    def construct_graph(self, is_train="train"):

        if is_train == "train":
            keep_prob = self.keep_prob
        else:
            keep_prob = 1.0

        task_infos = self.construct_task_outs(keep_prob)
        metrics_opts_names = task_infos[0]
        logits = task_infos[1]
        task_loss = task_infos[2]
        # print(task_loss)

        #
        with tf.variable_scope("outputs", reuse=tf.AUTO_REUSE):
            # num_metric_opts = int(len(metrics_opts_names) / 2)
            self.output = [logits]
        #
        if is_train == "train":
            ep_loss, variance_loss = self.construct_ep_outs(keep_prob=1.0, activation_fn=tf.sigmoid)
            with tf.name_scope("loss"):
                extra_regus = tf.reduce_sum(tf.reduce_mean(self.W_nbrs ** 2, axis=[1, 2]))
                for i in range(self.n_edge_type):
                    if self.edge_feats_lengths[i] > 1:
                        extra_regus += tf.reduce_mean(self.feat_parameters[i] ** 2)
                # variance_loss = tf.Print(variance_loss, [tf.shape(variance_loss)])
                global_loss = task_loss + self.alpha * ep_loss + self.beta * extra_regus + self.xi * variance_loss
                # global_loss = tf.Print(global_loss, [global_loss, ep_loss, extra_regus, task_loss, variance_loss])
            #     #
            with tf.variable_scope("summary", reuse=tf.AUTO_REUSE):
                n_metrics_opts = len(metrics_opts_names)
                summary_values = [global_loss, ep_loss, task_loss, extra_regus] + metrics_opts_names[
                                                                                  :int(
                                                                                      n_metrics_opts / 2)]
                # summary_names = ['loss_v', 'ep_loss_v', 'task_loss_v', 'extra_regus_v', 'variance_loss_v'] + metrics_opts_names[
                #                                                                                    int(n_metrics_opts / 2):]

                self.summary_names = ['loss_v', 'ep_loss_v', 'task_loss_v', 'extra_regus_v'] + metrics_opts_names[
                                                                                               int(n_metrics_opts / 2):]
                if self.sample_method == 'adaptive':
                    self.summary_names.append('variance_loss_v')
                    summary_values.append(variance_loss)

            with tf.name_scope("opt"):
                opt_optimize = self.optimizer.minimize(global_loss, global_step=self.global_step)

                self.opt_model = [opt_optimize, summary_values]
                # self.opt_model = [ep_loss]

    def batch_distance(self, embedding, nbr_embedding):
        distance = tf.reduce_mean(tf.square(embedding - nbr_embedding), axis=1)
        return distance

    def batch_distance_neg(self, embedding, nbr_embedding):
        red = nbr_embedding - tf.expand_dims(embedding, 1)
        distance = tf.reduce_mean(tf.square(red), axis=2)
        return distance

    def batch_distance_pair(self, pos, neg):
        return tf.reduce_sum(tf.nn.relu(tf.expand_dims(pos, 1) - neg + self.gamma), axis=1)
