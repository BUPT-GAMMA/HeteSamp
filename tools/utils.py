import numpy as np


def create_or_update_placeholder_bf(task_infos, ep_infos, model, task, n_edge_type, is_train, adaptive=False):
    info_dicts = {}
    if task == "MCP":
        nids, labels = task_infos
        info_dicts[model.nids] = nids
        info_dicts[model.labels] = labels
    elif task == "Regress":
        nids, labels = task_infos
        info_dicts[model.nids] = nids
        info_dicts[model.labels] = labels
    else:
        lids, rids, labels = task_infos
        info_dicts[model.lids] = lids
        info_dicts[model.rids] = rids
        info_dicts[model.labels] = labels

    if is_train == "train":

        info_dicts[model.nids_ep] = ep_infos[0]

        info_dicts[model.nids_types] = ep_infos[1]

        info_dicts[model.neg_ids] = ep_infos[2]

        info_dicts[model.seg_ids] = ep_infos[3]
        info_dicts[model.nbr_ids] = ep_infos[4]

        for i in range(n_edge_type):
            info_dicts[model.nbr_feats[i]] = ep_infos[5][i]

        info_dicts[model.edge_weight] = ep_infos[6]

        info_dicts[model.edge_types] = ep_infos[7]
        info_dicts[model.batch_size] = [ep_infos[8]]

        if adaptive:
            # info_dicts[model.count_samples] = ep_infos[9]
            info_dicts[model.edge_count] = ep_infos[9]
            # print(ep_infos[9])
    return info_dicts


def create_or_update_placeholder(task_infos, ep_infos, model, task, n_edge_type, is_train, adaptive=False):
    info_dicts = {}
    if task == "MCP":
        nids, labels = task_infos
        info_dicts[model.nids] = nids
        info_dicts[model.labels] = labels
    elif task == "Regress":
        nids, labels = task_infos
        info_dicts[model.nids] = nids
        info_dicts[model.labels] = labels
    else:
        lids, rids, labels = task_infos
        info_dicts[model.lids] = lids
        info_dicts[model.rids] = rids
        info_dicts[model.labels] = labels

    if is_train == "train":
        info_dicts[model.nids_ep] = ep_infos[0]
        info_dicts[model.nids_types] = ep_infos[1]
        info_dicts[model.neg_ids] = ep_infos[2]
        for i in range(n_edge_type):
            info_dicts[model.seg_ids[i]] = ep_infos[3][i]
            info_dicts[model.nbr_ids[i]] = ep_infos[4][i]
            info_dicts[model.nbr_feats[i]] = ep_infos[5][i]
            info_dicts[model.edge_weight[i]] = ep_infos[6][i]
            info_dicts[model.edge_types[i]] = ep_infos[7][i]
            if adaptive:
                info_dicts[model.edge_count[i]] = ep_infos[9][i]
        info_dicts[model.batch_size] = ep_infos[8]

    return info_dicts


def create_or_update_placeholder2(task_infos, model, task):
    info_dicts = {}
    if task == "MCP":
        nids, labels = task_infos
        info_dicts[model.nids] = nids
        info_dicts[model.labels] = labels
    elif task == "Regress":
        nids, labels = task_infos
        info_dicts[model.nids] = nids
        info_dicts[model.labels] = labels
    else:
        lids, rids, labels = task_infos
        info_dicts[model.lids] = lids
        info_dicts[model.rids] = rids
        info_dicts[model.labels] = labels
    return info_dicts


def summary_infos(summary, values, names):
    for i, name in enumerate(names):
        summary.value.add(tag=name, simple_value=values[i])

    return summary


def static_infos(ep_infos):
    node_numbers = len(ep_infos[0])
    edge_numbers = np.sum([len(ep_infos[3][i]) for i in range(len(ep_infos[3]))])
    nbr_numbers = np.sum([len(np.unique(ep_infos[3][i])) for i in range(len(ep_infos[3]))])
    return "{}\t{}\t{}\t".format(node_numbers, edge_numbers, nbr_numbers)
