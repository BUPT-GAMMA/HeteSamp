import tensorflow as tf

class HIG:
    # HIG (heterogeneous interaction graph) is to load information from files
    def __init__(self, data_config):
        self.data_config = data_config

    def load_label_infos(self, file_name, batch_size, num_epochs=1, min_after_times=32, capacity_times=64,
                         shuffle=True):
        col_names, col_defvals = build_task_cols(self.data_config["task"])
        input_file = self.data_config["data_path"] + file_name

        min_after_dequeue = (self.data_config["label_min_after_times"]
                             if "label_min_after_times" in self.data_config else min_after_times) * batch_size
        capacity = (self.data_config["label_capacity_times"]
                    if "label_capacity_times" in self.data_config else capacity_times) * batch_size
        filename_queue = tf.train.string_input_producer([input_file], num_epochs=num_epochs, shuffle=shuffle)
        reader = tf.TextLineReader()
        _, value = reader.read_up_to(filename_queue, batch_size)
        value = tf.train.shuffle_batch(
            [value],
            batch_size=batch_size,
            num_threads=24,
            capacity=capacity,
            enqueue_many=True,
            min_after_dequeue=min_after_dequeue)
        infos = tf.decode_csv(
            value, record_defaults=col_defvals, field_delim=self.data_config["label_info_delim"], use_quote_delim=False)
        return infos

    def load_embedding_propagation_info(self, file_name, batch_size, num_epochs=1, min_after_times=32,
                                        capacity_times=64, shuffle=True):

        num_edge_types = self.data_config["num_edge_types"]
        input_file = self.data_config["data_path"] + file_name

        col_names, col_defvals = build_EP_cols(num_edge_types)

        min_after_dequeue = (self.data_config["EP_min_after_times"]
                             if "EP_min_after_times" in self.data_config else min_after_times) * batch_size
        capacity = (self.data_config["EP_capacity_times"]
                    if "EP_capacity_times" in self.data_config else capacity_times) * batch_size

        reader = tf.TextLineReader()
        filename_queue = tf.train.string_input_producer([input_file], num_epochs=num_epochs, shuffle=shuffle)
        _, value = reader.read_up_to(filename_queue, batch_size)
        value = tf.train.shuffle_batch(
            [value],
            batch_size=batch_size,
            num_threads=24,
            capacity=capacity,
            enqueue_many=True,
            min_after_dequeue=min_after_dequeue)

        col_delim = self.data_config["EP_info_delim"]
        node_delim = self.data_config["EP_node_delim"]
        feat_delim = self.data_config["EP_feat_delim"]

        features = tf.decode_csv(value, record_defaults=col_defvals, field_delim=col_delim, use_quote_delim=False)

        nids = features[0]
        tids = features[1]
        neg_ids = extract_neg_nbrs(features[2], node_delim)

        nbr_ids = []
        nbr_feats = []
        nbr_segs = []

        edge_types = []
        for i in range(num_edge_types):
            nbr_seg, nbr_idx = extract_typed_nbrs(features[3 + i], node_delim)
            _, nbr_feat = extract_typed_nbrs_feats(features[3 + num_edge_types + i], node_delim, feat_delim)

            nbr_ids.append(nbr_idx)
            nbr_segs.append(nbr_seg)
            nbr_feats.append(nbr_feat)
            edge_types.append(features[3 + num_edge_types * 2 + i])

        return nids, tids, neg_ids, nbr_ids, nbr_segs, nbr_feats, edge_types


def build_task_cols(task="MCP"):
    if task == "MCP":  # multi-class classification
        col_names = ["lid", "label"]
        col_vals = [[-1], [-1]]
    elif task == "LP":  # link prediction
        col_names = ["lid", "rid", "label"]
        col_vals = [[-1], [-1], [-1]]
    elif task == "Regress":  # rating prediction and other regression
        col_names = ["lid", "rid", "label"]
        col_vals = [[-1], [-1], [-1.0]]
    else:
        col_names = ["lid", "label"]
        col_vals = [[-1], [-1]]
    return col_names, col_vals


def build_EP_cols(num_edge_types):
    col_names = ['nid', 'tid', 'neg']
    col_vals = [[-1], [-1], ['']]
    col_names += ['nbr_ids_{}'.format(i) for i in range(num_edge_types)]
    col_vals += [['']] * num_edge_types
    col_names += ['nbr_edges_{}'.format(i) for i in range(num_edge_types)]
    col_vals += [['']] * num_edge_types
    col_names += ['nbr_tids_{}'.format(i) for i in range(num_edge_types)]
    col_vals += [[-1]] * num_edge_types
    return col_names, col_vals


def extract_neg_nbrs(nbr_list_str, node_delim):
    nbr_ids_sparse = tf.string_split(nbr_list_str, node_delim)
    nbr_ids_matrix = tf.sparse_to_dense(
        sparse_indices=nbr_ids_sparse.indices,
        output_shape=nbr_ids_sparse.dense_shape,
        sparse_values=tf.string_to_number(nbr_ids_sparse.values, out_type=tf.int32),
    )
    return nbr_ids_matrix


def extract_typed_nbrs(nbr_list_str, node_delim):
    nbr_ids_sparse = tf.string_split(nbr_list_str, node_delim)
    seg_indices = nbr_ids_sparse.indices[:, 0]
    nbr_ids = tf.string_to_number(nbr_ids_sparse.values, out_type=tf.int32)
    return seg_indices, nbr_ids

def extract_typed_nbrs_feats(nbr_list_str, node_delim, feat_delim):
    nbr_edges_sparse = tf.string_split(nbr_list_str, node_delim)
    seg_indices = nbr_edges_sparse.indices[:, 0]
    nbr_edge_elements = tf.string_split(nbr_edges_sparse.values, feat_delim)
    nbr_edge_features = tf.sparse_to_dense(
        sparse_indices=nbr_edge_elements.indices,
        output_shape=nbr_edge_elements.dense_shape,
        sparse_values=tf.string_to_number(nbr_edge_elements.values, out_type=tf.float32),
    )
    return seg_indices, nbr_edge_features
