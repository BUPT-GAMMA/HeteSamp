# for fast_sampler, we only pay attention to node weight/ numerical feature
import tensorflow as tf

from fast_samper import FastSamper


class RandomSamper(FastSamper):

    def __init__(self, sampler_type, sampler_num, seed, self_normalize='False', sampler_num_balance=True):
        super(RandomSamper, self).__init__(sampler_type, sampler_num, seed, self_normalize, sampler_num_balance)

    def edge_weight_sampler_type_dependent(self, seg_ids_type, nbr_ids_type, nbr_feats_type, p_type,
                                           num_sample_type, edge_type_id=-1):
        nbr_real_ids, nbr_relative_ids = tf.unique(nbr_ids_type)
        nbr_counts = tf.size(nbr_real_ids)
        nbr_length = tf.size(nbr_relative_ids)

        unnormal_q = tf.ones(shape=[nbr_counts, ], dtype=tf.float32)

        sampled_ids = \
            tf.multinomial(tf.log(tf.expand_dims(tf.reshape(unnormal_q, (-1,)), axis=0)), num_samples=num_sample_type)[
                0]

        unique_sample_ids, _, unique_sample_counts = tf.unique_with_counts(sampled_ids)

        # select the position of sampled nbr_ids
        partitions = tf.gather(
            tf.unsorted_segment_sum(tf.ones_like(unique_sample_ids, tf.int32), unique_sample_ids,
                                    nbr_counts), nbr_relative_ids)

        sampled_indices = tf.cast(tf.dynamic_partition(tf.range(nbr_length), partitions, 2)[1], tf.int32)

        sampled_segs = tf.cast(tf.gather(seg_ids_type, sampled_indices), tf.int32)
        sampled_nbrs = tf.gather(nbr_ids_type, sampled_indices)
        sampled_feats = tf.gather(nbr_feats_type, sampled_indices)

        unique_sampled_seg_y, unique_sampled_seg_idx, unique_sampled_seg_counts = tf.unique_with_counts(sampled_segs)

        # v1
        # sampled_weight = tf.gather(1.0 / tf.cast(unique_sampled_seg_counts, tf.float32), unique_sampled_seg_idx)

        # v2

        sampled_p = tf.gather(p_type, sampled_indices)
        count_samples = tf.cast(tf.unsorted_segment_sum(unique_sample_counts, unique_sample_ids, nbr_counts),
                                tf.float32)
        sampled_relative_ids = tf.gather(nbr_relative_ids, sampled_indices)

        # sampled_weight = sampled_p * tf.gather(tf.reshape(count_samples, (-1,)), sampled_relative_ids) / (tf.cast(
        #     num_sample_type, tf.float32) * tf.cast(nbr_counts, tf.float32))
        #
        # v3
        sampled_weight = sampled_p * tf.gather(tf.reshape(count_samples, (-1,)), sampled_relative_ids) / (
            tf.cast(num_sample_type, tf.float32))

        return [sampled_segs, sampled_nbrs, sampled_feats, sampled_weight]

    # -----------------------------type fusion----------------------------------------

    def edge_weight_sampler_type_fusion(self, seg_ids, nbr_ids, nbr_feats, norm_edge_weight, num_samples,
                                        partition_edge_type, partition_edge_range):
        p_j_i_r = norm_edge_weight

        nbr_real_ids, nbr_relative_ids = tf.unique(nbr_ids)
        nbr_counts = tf.size(nbr_real_ids)
        nbr_length = tf.size(nbr_relative_ids)

        unnormal_q = tf.ones(shape=[nbr_counts, ], dtype=tf.float32)
        norm_q = unnormal_q / tf.reduce_sum(unnormal_q)

        sampled_ids = \
            tf.multinomial(tf.log(tf.expand_dims(tf.reshape(unnormal_q, (-1,)), axis=0)), num_samples=num_samples)[0]

        unique_sample_ids, _, unique_sample_counts = tf.unique_with_counts(sampled_ids)
        # unique_sample_counts = tf.Print(unique_sample_counts, [unique_sample_counts, unique_sample_ids])
        count_samples = tf.cast(tf.unsorted_segment_sum(unique_sample_counts, unique_sample_ids, nbr_counts),
                                tf.float32)
        # select the position of sampled nbr_ids
        partitions = tf.gather(
            tf.unsorted_segment_sum(tf.ones_like(unique_sample_ids, tf.int32), unique_sample_ids,
                                    nbr_counts), nbr_relative_ids)

        # the sampled infos

        sampled_indices = tf.cast(tf.dynamic_partition(tf.range(nbr_length), partitions, 2)[1], tf.int32)

        sampled_relative_ids = tf.gather(nbr_relative_ids, sampled_indices)
        sampled_q = tf.gather(tf.reshape(count_samples, (-1,)) / norm_q, sampled_relative_ids)
        sampled_p = tf.gather(p_j_i_r, sampled_indices)
        # sampled_q = tf.Print(sampled_q, [tf.shape(sampled_p), tf.shape(sampled_q), "shape_p and shape_q"])

        sampled_segs = tf.cast(tf.gather(seg_ids, sampled_indices), tf.int32)
        sampled_nbrs = tf.gather(nbr_ids, sampled_indices)
        # sampled_feats = tf.gather(nbr_feats, sampled_indices)
        sampled_feats = []
        # sampled_indices = tf.Print(sampled_indices, [sampled_indices])
        sampled_feat_indices = tf.gather(partition_edge_range, sampled_indices)
        sampled_feat_types = tf.gather(partition_edge_type, sampled_indices)

        n_edge_type = len(nbr_feats)

        sampled_feat_indices_list = tf.dynamic_partition(sampled_feat_indices, sampled_feat_types, n_edge_type)

        for i in range(n_edge_type):
            sampled_feats.append(
                tf.cond(tf.greater(tf.size(sampled_feat_indices_list[i]), 0),
                        true_fn=lambda: tf.gather(nbr_feats[i], sampled_feat_indices_list[i]),
                        false_fn=lambda: tf.Variable([[]], tf.float32)
                        )
            )

        unique_sampled_seg_y, unique_sampled_seg_idx, unique_sampled_seg_counts = tf.unique_with_counts(sampled_segs)

        sampled_weight = sampled_p * tf.gather(tf.reshape(count_samples, (-1,)), sampled_relative_ids) / (
            tf.cast(num_samples, tf.float32))

        sampled_nbrs_list = tf.dynamic_partition(sampled_nbrs, sampled_feat_types, n_edge_type)
        sampled_weight_list = tf.dynamic_partition(sampled_weight, sampled_feat_types, n_edge_type)
        sampled_feat_types_list = tf.dynamic_partition(sampled_feat_types, sampled_feat_types, n_edge_type)
        sampled_segs_list = tf.dynamic_partition(sampled_segs, sampled_feat_types, n_edge_type)

        return [sampled_segs_list, sampled_nbrs_list, sampled_feats, sampled_weight_list, sampled_feat_types_list]
