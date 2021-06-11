# for fast_sampler, we only pay attention to node weight/ numerical feature
import tensorflow as tf


class FastSamper:

    def __init__(self, sampler_type, sampler_num, seed, self_normalize='False', sampler_num_balance=1):

        self.sampler_type = sampler_type
        self.sampler_num = sampler_num
        self.sampler_num_balance = sampler_num_balance
        self.seed = seed
        self.self_normalize = True if self_normalize == 'True' else False

    def sampler(self, candidate_infos, feat_parameters=None, utilize_edge_feats=True):
        if self.sampler_type == "TDSampler":
            return self.type_dependent_sampler(candidate_infos, utilize_edge_feats)
        elif self.sampler_type == 'TFSampler':
            return self.type_fusion_sampler(candidate_infos, utilize_edge_feats)
        else:
            print("Sampler Error: please select TDSampler or TFSampler")
            exit()

    def type_dependent_sampler(self, candidate_infos, utilize_edge_feats=True):

        # utilize_edge_feats reequires parameters to learn weight
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

        sampled_nbr_ids = []
        sampled_nbr_segs = []
        sampled_nbr_feats = []
        sampled_edge_weight = []

        # for type-dependent sampler, we calculate the number of candidates for each type.
        num_node_types = []  # the number of sampled nodes
        num_nodes_all = 0
        for i in range(n_edge_types):
            num_node_type_i = tf.cast(tf.size(tf.unique(nbr_ids[i]).y), tf.float32)
            num_node_types.append(num_node_type_i)
            num_nodes_all += num_node_type_i

        num_sampled_nodes = []
        for i in range(n_edge_types):
            num_sampled_nodes.append(
                tf.cast(tf.ceil(self.sampler_num * num_node_types[i] / num_nodes_all), tf.int32)
            )

        for i in range(n_edge_types):
            sampled_info_list = tf.cond(
                tf.less(num_sampled_nodes[i], 1),
                true_fn=lambda: [tf.Variable([], dtype=tf.int32), tf.Variable([], dtype=tf.int32),
                                 tf.Variable([[]], dtype=tf.float32), tf.Variable([], dtype=tf.float32)],
                false_fn=lambda: self.edge_weight_sampler_type_dependent(nbr_segs[i], nbr_ids[i], nbr_feats[i],
                                                                         p_all[i], num_sampled_nodes[i], i)
            )

            sampled_nbr_segs.append(sampled_info_list[0])
            sampled_nbr_ids.append(sampled_info_list[1])
            sampled_nbr_feats.append(sampled_info_list[2])
            sampled_edge_weight.append(sampled_info_list[3])

        return self.extract_final_data_type_dependent(nids, tids, neg_ids, sampled_nbr_segs, sampled_nbr_ids,
                                                      sampled_nbr_feats, sampled_edge_weight)

    def edge_weight_sampler_type_dependent(self, seg_ids_type, nbr_ids_type, nbr_feats_type, p_type,
                                           num_sample_type, edge_type_id=-1):

        nbr_real_ids, nbr_relative_ids = tf.unique(nbr_ids_type)
        nbr_counts = tf.size(nbr_real_ids)
        nbr_length = tf.size(nbr_relative_ids)

        unnormal_q = tf.unsorted_segment_sum(p_type ** 2, nbr_relative_ids, nbr_counts)
        norm_q = unnormal_q / tf.reduce_sum(unnormal_q)

        sampled_ids = \
            tf.multinomial(tf.log(tf.expand_dims(tf.reshape(unnormal_q, (-1,)), axis=0)), num_samples=num_sample_type)[
                0]

        unique_sample_ids, _, unique_sample_counts = tf.unique_with_counts(sampled_ids)

        count_samples = tf.cast(tf.unsorted_segment_sum(unique_sample_counts, unique_sample_ids, nbr_counts),
                                tf.float32)

        partitions = tf.gather(
            tf.unsorted_segment_sum(tf.ones_like(unique_sample_ids, tf.int32), unique_sample_ids,
                                    nbr_counts), nbr_relative_ids)

        sampled_indices = tf.cast(tf.dynamic_partition(tf.range(nbr_length), partitions, 2)[1], tf.int32)

        sampled_relative_ids = tf.gather(nbr_relative_ids, sampled_indices)
        sampled_q = tf.gather(tf.reshape(count_samples, (-1,)) / norm_q, sampled_relative_ids)
        sampled_p = tf.gather(p_type, sampled_indices)

        sampled_segs = tf.cast(tf.gather(seg_ids_type, sampled_indices), tf.int32)
        sampled_nbrs = tf.gather(nbr_ids_type, sampled_indices)
        sampled_feats = tf.gather(nbr_feats_type, sampled_indices)

        sampled_weight = sampled_p * sampled_q / tf.cast(num_sample_type, tf.float32)

        if self.self_normalize:
            sampled_q = tf.gather(norm_q, sampled_relative_ids)
            ratio = sampled_p / sampled_q
            sampled_weight = ratio / tf.gather(tf.segment_sum(ratio, sampled_segs), sampled_segs)

        return [sampled_segs, sampled_nbrs, sampled_feats, sampled_weight]

    def extract_final_data_type_dependent(self, nids, tids, neg_ids, sampled_nbr_segs, sampled_nbr_ids,
                                          sampled_nbr_feats,
                                          sampled_edge_weight):

        # actually, we need to sample nids and rewrite nbr_segs
        n_edge_type = len(sampled_nbr_ids)
        sampled_nids_indices, sampled_nids_map_ids = tf.unique(tf.concat(sampled_nbr_segs, axis=0))

        partitions = tf.concat(
            [tf.cast(tf.ones_like(sampled_nbr_segs[i]) * i, tf.int32) for i in range(len(sampled_nbr_segs))], axis=0)

        final_sampled_nids = tf.gather(nids, sampled_nids_indices)
        final_sampled_tids = tf.gather(tids, sampled_nids_indices)
        final_sampled_neg_ids = tf.gather(neg_ids, sampled_nids_indices)
        final_batch_size = tf.size(sampled_nids_indices)

        final_edge_types = [tf.ones(tf.size(sampled_nbr_ids[i])) * i for i in range(n_edge_type)]

        final_sampled_nbr_segs = tf.dynamic_partition(sampled_nids_map_ids, partitions, len(sampled_nbr_segs))

        final_sampled_nbr_ids = sampled_nbr_ids
        final_sampled_nbr_feats = sampled_nbr_feats
        final_sampled_edge_weight = sampled_edge_weight

        return final_sampled_nids, final_sampled_tids, final_sampled_neg_ids, final_sampled_nbr_segs, \
               final_sampled_nbr_ids, final_sampled_nbr_feats, final_sampled_edge_weight, \
               final_edge_types, final_batch_size

    # -----------------------------type fusion----------------------------------------
    def type_fusion_sampler(self, candidate_infos, utilize_edge_feats=True):

        nids, tids, neg_ids, nbr_ids, nbr_segs, nbr_feats, edge_types = candidate_infos
        n_edge_types = len(nbr_feats)

        # norm_edge_weight = []
        # edge_nums = [tf.size(nbr_ids[i]) for i in range(n_edge_types)]

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
        global_p_all = tf.concat(p_all, axis=0)

        partition_edge_range = tf.concat([tf.range(tf.size(nbr_ids[i])) for i in range(n_edge_types)], axis=0)
        partition_edge_type = tf.concat([tf.ones(tf.size(nbr_ids[i]), tf.int32) * i for i in range(n_edge_types)],
                                        axis=0)

        # partition_edge_type = tf.Print(partition_edge_type, [partition_edge_type, "partition_edge_type"])
        sampled_infos = self.edge_weight_sampler_type_fusion(global_segs_ids, global_nbr_ids, global_nbr_feat,
                                                             global_p_all, self.sampler_num,
                                                             partition_edge_type, partition_edge_range)
        sampled_nbr_segs = sampled_infos[0]
        sampled_nbr_ids = sampled_infos[1]
        sampled_nbr_feats = sampled_infos[2]
        sampled_edge_weight = sampled_infos[3]
        sampled_edge_type = sampled_infos[4]

        return self.extract_final_data_type_fusion(nids, tids, neg_ids, sampled_nbr_segs, sampled_nbr_ids,
                                                   sampled_nbr_feats, sampled_edge_weight, sampled_edge_type)

    def edge_weight_sampler_type_fusion(self, seg_ids, nbr_ids, nbr_feats, norm_edge_weight, num_samples,
                                        partition_edge_type, partition_edge_range):

        p_j_i_r = norm_edge_weight

        nbr_real_ids, nbr_relative_ids = tf.unique(nbr_ids)
        nbr_counts = tf.size(nbr_real_ids)
        nbr_length = tf.size(nbr_relative_ids)

        unnormal_q = tf.unsorted_segment_sum(p_j_i_r ** 2, nbr_relative_ids, nbr_counts)
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

        sampled_weight = sampled_p * sampled_q / tf.cast(num_samples, tf.float32)

        unique_sampled_seg_y, unique_sampled_seg_idx = tf.unique(sampled_segs)

        if self.self_normalize:
            sampled_q = tf.gather(norm_q, sampled_relative_ids)
            ratio = sampled_p / sampled_q
            sampled_weight = ratio / tf.gather(
                tf.unsorted_segment_sum(ratio, unique_sampled_seg_idx, tf.size(unique_sampled_seg_y)),
                unique_sampled_seg_idx)

        sampled_nbrs_list = tf.dynamic_partition(sampled_nbrs, sampled_feat_types, n_edge_type)
        sampled_weight_list = tf.dynamic_partition(sampled_weight, sampled_feat_types, n_edge_type)
        sampled_feat_types_list = tf.dynamic_partition(sampled_feat_types, sampled_feat_types, n_edge_type)
        sampled_segs_list = tf.dynamic_partition(sampled_segs, sampled_feat_types, n_edge_type)

        return [sampled_segs_list, sampled_nbrs_list, sampled_feats, sampled_weight_list, sampled_feat_types_list]

    def extract_final_data_type_fusion(self, nids, tids, neg_ids, sampled_nbr_segs, sampled_nbr_ids, sampled_nbr_feats,
                                       sampled_edge_weight, sampled_feat_types):

        # actually, we need to sample nids and rewrite nbr_segs
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

        final_batch_size = tf.size(sampled_nids_indices)
        final_edge_types = sampled_feat_types

        return final_sampled_nids, final_sampled_tids, final_sampled_neg_ids, final_sampled_nbr_segs, \
               final_sampled_nbr_ids, final_sampled_nbr_feats, final_sampled_edge_weight, final_edge_types, \
               final_batch_size
