# for fast_sampler, we only pay attention to node weight/ numerical feature
import tensorflow as tf

from fast_samper import FastSamper


class AdaptSampler(FastSamper):

    def __init__(self, sampler_type, sampler_num, seed, self_normalize=False, sampler_num_balance=True):
        super(AdaptSampler, self).__init__(sampler_type, sampler_num, seed, self_normalize, sampler_num_balance)

    # -----------------------------type fusion----------------------------------------
    def sampler(self, candidate_infos, feat_parameters=None, utilize_edge_feats=True):

        nids, tids, neg_ids, nbr_ids, nbr_segs, nbr_feats, edge_types = candidate_infos
        n_edge_types = len(nbr_feats)
        edge_nums = [tf.size(nbr_ids[i]) for i in range(n_edge_types)]
        p_all = []
        for i in range(n_edge_types):
            nbr_seg_i = nbr_segs[i]

            p_i = tf.cond(
                tf.less(edge_nums[i], 1),
                true_fn=lambda: tf.Variable([], dtype=tf.float32),
                false_fn=lambda: tf.ones_like(nbr_seg_i, tf.float32) / tf.gather(
                    tf.segment_sum(tf.ones_like(nbr_seg_i, tf.float32), nbr_seg_i), nbr_seg_i)
            )
            p_all.append(p_i)

        global_segs_ids = tf.concat(nbr_segs, axis=0)
        global_nbr_ids = tf.concat(nbr_ids, axis=0)
        global_nbr_feat = nbr_feats
        global_norm_edge_weight = tf.concat(p_all, axis=0)

        partition_edge_range = tf.concat([tf.range(tf.size(nbr_ids[i])) for i in range(n_edge_types)], axis=0)
        partition_edge_type = tf.concat([tf.ones(tf.size(nbr_ids[i]), tf.int32) * i for i in range(n_edge_types)],
                                        axis=0)

        sampled_infos = self.adaptive_sampler_edge_weight(global_segs_ids, global_nbr_ids, global_nbr_feat,
                                                          global_norm_edge_weight, self.sampler_num,
                                                          partition_edge_type, partition_edge_range, feat_parameters)
        sampled_nbr_segs = sampled_infos[0]
        sampled_nbr_ids = sampled_infos[1]
        sampled_nbr_feats = sampled_infos[2]
        sampled_edge_weight = sampled_infos[3]
        sampled_edge_type = sampled_infos[4]
        sampled_samples = sampled_infos[5]

        return self.extract_final_data_adaptive(nids, tids, neg_ids, sampled_nbr_segs, sampled_nbr_ids,
                                                sampled_nbr_feats, sampled_edge_weight, sampled_edge_type,
                                                sampled_samples)

    def adaptive_sampler_edge_weight(self, seg_ids, nbr_ids, nbr_feats, nbr_link_weight, num_samples,
                                     partition_edge_type, partition_edge_range, feat_parameters):

        p_j_i_r = nbr_link_weight
        n_edge_type = len(nbr_feats)

        att_edge = []
        for i in range(n_edge_type):
            # nbr_feats[i] = tf.Print(nbr_feats[i], [tf.size(nbr_feats[i]), tf.size(p_j_i_r), i, "weight"])
            att_edge.append(
                tf.cond(tf.reduce_any(tf.equal(partition_edge_type, i)),
                        false_fn=lambda: tf.Variable([], tf.float32),
                        true_fn=lambda: tf.reshape(
                            tf.clip_by_value(tf.nn.sigmoid(tf.matmul(nbr_feats[i], feat_parameters[i])), 1e-6,
                                             1 - 1e-6), (-1,))
                        )
            )

        att_edge = tf.concat(att_edge, axis=0)

        nbr_real_ids, nbr_relative_ids = tf.unique(nbr_ids)
        nbr_counts = tf.size(nbr_real_ids)
        nbr_length = tf.size(nbr_relative_ids)

        unnormal_q = tf.unsorted_segment_sum(tf.reshape(att_edge, (-1,)) * p_j_i_r, nbr_relative_ids, nbr_counts)
        norm_q = unnormal_q / tf.reduce_sum(unnormal_q)

        sampled_ids = \
            tf.multinomial(tf.log(tf.expand_dims(tf.reshape(unnormal_q, (-1,)), axis=0)), num_samples=num_samples,
                           seed=self.seed)[0]

        unique_sample_ids, _, unique_sample_counts = tf.unique_with_counts(sampled_ids)

        count_samples = tf.cast(tf.unsorted_segment_sum(unique_sample_counts, unique_sample_ids, nbr_counts),
                                tf.float32)

        # select the position of sampled nbr_ids

        # unique_sample_ids = tf.Print(unique_sample_ids, [tf.size(unique_sample_ids), nbr_counts, "unique_sampled_ids"])
        partitions = tf.gather(
            tf.unsorted_segment_sum(tf.ones_like(unique_sample_ids, tf.int32), unique_sample_ids, nbr_counts),
            nbr_relative_ids)
        # the sampled infos
        sampled_indices = tf.cast(tf.dynamic_partition(tf.range(nbr_length), partitions, 2)[1], tf.int32)

        sampled_relative_ids = tf.gather(nbr_relative_ids, sampled_indices)
        sampled_q = tf.gather(norm_q, sampled_relative_ids)
        sampled_p = tf.gather(p_j_i_r, sampled_indices)
        # sampled_q = tf.Print(sampled_q, [tf.shape(sampled_p), tf.shape(sampled_q), "shape_p and shape_q"])

        sampled_samples = tf.reshape(tf.gather(tf.reshape(count_samples, [-1, 1]), sampled_relative_ids), (-1,))

        sampled_segs = tf.cast(tf.gather(seg_ids, sampled_indices), tf.int32)
        sampled_nbrs = tf.gather(nbr_ids, sampled_indices)
        # sampled_feats = tf.gather(nbr_feats, sampled_indices)
        sampled_feats = []
        sampled_feat_indices = tf.gather(partition_edge_range, sampled_indices)
        sampled_feat_types = tf.gather(partition_edge_type, sampled_indices)
        sampled_feat_indices_list = tf.dynamic_partition(sampled_feat_indices, sampled_feat_types, n_edge_type)

        for i in range(n_edge_type):
            # sampled_feat_indices_list[i] = tf.Print(sampled_feat_indices_list[i],
            #                                         [tf.shape(sampled_feat_indices_list[i]), "list"])
            sampled_feats.append(
                tf.cond(tf.greater(tf.size(sampled_feat_indices_list[i]), 0),
                        true_fn=lambda: tf.gather(nbr_feats[i], sampled_feat_indices_list[i]),
                        false_fn=lambda: tf.Variable([[]], tf.float32)
                        )
            )

        types_sampled_segs_list = tf.dynamic_partition(sampled_segs, sampled_feat_types, n_edge_type)
        types_sampled_nbrs_list = tf.dynamic_partition(sampled_nbrs, sampled_feat_types, n_edge_type)
        sampled_weight_list = tf.dynamic_partition(sampled_p / sampled_q, sampled_feat_types, n_edge_type)
        sampled_feat_types_list = tf.dynamic_partition(sampled_feat_types, sampled_feat_types, n_edge_type)
        sampled_samples_list = tf.dynamic_partition(sampled_samples, sampled_feat_types, n_edge_type)

        return [types_sampled_segs_list, types_sampled_nbrs_list, sampled_feats, sampled_weight_list,
                sampled_feat_types_list, sampled_samples_list]


    def extract_final_data_adaptive(self, nids, tids, neg_ids, sampled_nbr_segs, sampled_nbr_ids, sampled_nbr_feats,
                                    sampled_edge_weight, sampled_feat_types, sampled_samples):

        sampled_nids_indices, sampled_nids_map_ids = tf.unique(tf.concat(sampled_nbr_segs, axis=0))

        final_sampled_nids = tf.gather(nids, sampled_nids_indices)
        final_sampled_tids = tf.gather(tids, sampled_nids_indices)
        final_sampled_neg_ids = tf.gather(neg_ids, sampled_nids_indices)

        partitions = tf.concat(
            [tf.cast(tf.ones_like(sampled_nbr_segs[i]) * i, tf.int32) for i in range(len(sampled_nbr_segs))], axis=0)
        final_sampled_nbr_segs = tf.dynamic_partition(sampled_nids_map_ids, partitions, len(sampled_nbr_segs))

        final_sampled_nbr_ids = sampled_nbr_ids
        final_sampled_nbr_feats = sampled_nbr_feats
        final_sampled_edge_weight = sampled_edge_weight
        final_sampled_samples = sampled_samples

        final_batch_size = tf.size(sampled_nids_indices)
        final_edge_types = sampled_feat_types

        return final_sampled_nids, final_sampled_tids, final_sampled_neg_ids, final_sampled_nbr_segs, \
               final_sampled_nbr_ids, final_sampled_nbr_feats, final_sampled_edge_weight, final_edge_types, \
               final_batch_size, final_sampled_samples
