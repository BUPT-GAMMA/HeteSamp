# for fast_sampler, we only pay attention to node weight/ numerical feature
import tensorflow as tf


class General:

    def __init__(self, sampler_type, sampler_num, seed, self_normalize='False', sampler_num_balance=True):
        self.sampler_type = sampler_type
        self.sampler_num = sampler_num
        self.sampler_num_balance = sampler_num_balance

    def sampler(self, candidate_infos, parameters=None):
        nids, tids, neg_ids, nbr_ids, nbr_segs, nbr_feats, edge_types = candidate_infos
        n_edge_types = len(nbr_feats)
        norm_edge_weight = []
        edge_nums = [tf.size(nbr_ids[i]) for i in range(n_edge_types)]
        for i in range(n_edge_types):
            norm_edge_weight_i = tf.cond(
                tf.less(edge_nums[i], 1),
                true_fn=lambda: tf.Variable([], dtype=tf.float32),
                false_fn=lambda: tf.ones_like(nbr_segs[i], tf.float32) / tf.gather(
                    tf.segment_sum(tf.ones_like(nbr_segs[i], tf.float32), nbr_segs[i]),
                    nbr_segs[i])
            )
            norm_edge_weight.append(norm_edge_weight_i)

        return self.extract_final_data_non_sampler(nids, tids, neg_ids, nbr_segs, nbr_ids, nbr_feats, norm_edge_weight)

    def extract_final_data_non_sampler(self, nids, tids, neg_ids, nbr_segs, nbr_ids, nbr_feats, edge_weight):
        n_edge_type = len(nbr_ids)
        final_batch_size = tf.size(nids)
        final_edge_types = [tf.ones_like(nbr_ids[i]) * i for i in range(n_edge_type)]
        final_sampled_nbr_segs = nbr_segs
        final_sampled_nbr_ids = nbr_ids
        final_sampled_nbr_feats = nbr_feats
        final_sampled_edge_weight = edge_weight

        return nids, tids, neg_ids, final_sampled_nbr_segs, final_sampled_nbr_ids, final_sampled_nbr_feats, \
               final_sampled_edge_weight, final_edge_types, final_batch_size
