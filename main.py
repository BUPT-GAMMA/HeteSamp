import datetime
import os
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from adaptive_samper import AdaptSampler
from non_sampler import *
from random_sampler import *
from fast_samper import FastSamper

from configures.data_configures import *
from evulations import *
from input import HIG
from model import HeteSamp

from tools.path import *
from tools.utils import *

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'aminer', 'dataset name')
flags.DEFINE_float('learning_rate', 0.001, 'initial learning rate.')
flags.DEFINE_string('learning_algo', 'adam', '')

flags.DEFINE_string('model_path', '../model/', 'model path')
flags.DEFINE_string("is_train", "train", "is train / test")
# imdb
# flags.DEFINE_float("alpha", 1, "alpha")
# flags.DEFINE_float("beta", 0.1, "beta")
flags.DEFINE_float("alpha", 0.4, "alpha")
flags.DEFINE_float("beta", 0.01, "beta")
flags.DEFINE_float("gamma", 0.1, "gamma")
flags.DEFINE_float("psi", 0.1, "psi")
flags.DEFINE_float("xi", 0.5, "xi")
flags.DEFINE_integer("seed", 1, "random seed")
flags.DEFINE_string('summary', 'True', 'summary')

flags.DEFINE_integer('epochs', 10, 'num of epochs')
flags.DEFINE_integer('batch_size', 1024, 'size of batches')
flags.DEFINE_integer('test_batch_size', 128, 'test batch size')

flags.DEFINE_integer('embedding_dim', 128, 'embedding_dim')
flags.DEFINE_integer('early_stop', -1, 'early stop')

flags.DEFINE_float("keep_prob", 1.0, "keep_prob")

# sampler setting
flags.DEFINE_string('sample_method', 'fast', 'sample method')
flags.DEFINE_string('sample_type', 'NonSampler', 'type of samples')
flags.DEFINE_integer("num_sample", 32, 'number samples')
flags.DEFINE_integer("sampler_num_balance", 1, 'sampler number balance')
flags.DEFINE_string('self_normalize', 'False', 'self_normalize')
flags.DEFINE_string('variance_reduction', 'False', 'variance_reduction')

# flags.DEFINE_boolean('summary', True, 'summary')

# tf.set_random_seed(FLAGS.seed)
# np.random.seed(FLAGS.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def train_fast(model_path, model, configs, loader):
    create_directory(model_path)

    batch_size_label = int((configs["num_train_pairs"] * 1.0 / configs["num_nodes"]) * FLAGS.batch_size)
    batch_size_ep = FLAGS.batch_size

    inputs_task = loader.load_label_infos(configs["train_name"], batch_size_label, FLAGS.epochs)
    inputs_ep = loader.load_embedding_propagation_info(configs["graph_name"], batch_size_ep, FLAGS.epochs)

    global_step = tf.train.get_or_create_global_step()

    train_model = model(FLAGS, configs, global_step)
    train_model.construct_graph("train")
    if FLAGS.sample_method == "fast":
        sampler_model = FastSamper(FLAGS.sample_type, FLAGS.num_sample, FLAGS.seed, FLAGS.self_normalize,
                                   FLAGS.sampler_num_balance)
        feat_parameters = None
    elif FLAGS.sample_method == "random":
        sampler_model = RandomSamper(FLAGS.sample_type, FLAGS.num_sample, FLAGS.seed, FLAGS.self_normalize,
                                     FLAGS.sampler_num_balance)
        feat_parameters = None

    elif FLAGS.sample_method == 'adaptive':
        sampler_model = AdaptSampler(FLAGS.sample_type, FLAGS.num_sample, FLAGS.seed, FLAGS.self_normalize,
                                     FLAGS.sampler_num_balance)

        feat_parameters = train_model.feat_parameters

    else:
        sampler_model = General(FLAGS.sample_type, FLAGS.num_sample, FLAGS.seed, FLAGS.self_normalize,
                                FLAGS.sampler_num_balance)
        feat_parameters = None

    sampled_ep_infos = sampler_model.sampler(inputs_ep, feat_parameters)

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()

    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    bn_moving_vars += [g for g in g_list if 'global_step' in g.name]
    saver = tf.train.Saver(var_list=var_list + bn_moving_vars, max_to_keep=1)
    summary_names = train_model.summary_names
    hooks = [tf.train.StopAtStepHook(last_step=8000000000)]
    scaffold = tf.train.Scaffold(saver=saver, init_op=tf.global_variables_initializer())
    wf_times = open(model_path + "times.txt", "w")

    all_steps = int(FLAGS.epochs * configs['num_nodes'] * 1.0 / FLAGS.batch_size_ep)
    process = tqdm(total=all_steps)

    with tf.train.MonitoredTrainingSession(scaffold=scaffold, checkpoint_dir=model_path, save_checkpoint_secs=3600,
                                           hooks=hooks) as mon_sess:
        step = 0
        if FLAGS.summary:
            writer = tf.summary.FileWriter(model_path + "logs", mon_sess.graph)
            summary = tf.Summary()
            time_cost = 0
        try:
            flag_adaptive = (FLAGS.sample_method == 'adaptive')
            while not mon_sess.should_stop():
                step += 1
                if step > FLAGS.early_stop and FLAGS.early_stop > 0:
                    break
                if step % 200 == 0:
                    process.update(200)
                t1 = datetime.datetime.now()
                format_data_infos = mon_sess.run([inputs_task, sampled_ep_infos])

                data_dicts = create_or_update_placeholder(format_data_infos[0], format_data_infos[1], train_model,
                                                          configs["task"], configs["num_edge_types"], "train",
                                                          adaptive=flag_adaptive)
                t2 = datetime.datetime.now()

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                t3 = datetime.datetime.now()

                out_infos = mon_sess.run(train_model.opt_model, options=run_options, run_metadata=run_metadata,
                                         feed_dict=data_dicts)

                t4 = datetime.datetime.now()
                time_cost += (t4 - t1).total_seconds()

                if FLAGS.summary == 'True':
                    writer.add_run_metadata(run_metadata, 'step%d' % step)
                    summary = summary_infos(summary, out_infos[1], summary_names)
                    writer.add_summary(summary, step)
                    wf_times.write(
                        "{}\t{}\t{}\t{}\n".format(time_cost, (t2 - t1).total_seconds(), (t4 - t3).total_seconds(),
                                                  static_infos(format_data_infos[1])))
        finally:
            process.close()
            wf_times.close()
            writer.close()


def test(model_path, model, file_name, output_file, configs, loader):
    batch_size_lp = FLAGS.test_batch_size
    inputs_task = loader.load_label_infos(file_name, batch_size_lp, 1)

    global_step = tf.train.get_or_create_global_step()
    test_model = model(FLAGS, configs, global_step)

    test_model.construct_graph(is_train="test")
    saver = tf.train.Saver()
    hooks = [tf.train.StopAtStepHook(last_step=8000000000)]
    with tf.train.MonitoredTrainingSession(scaffold=tf.train.Scaffold(saver=saver), hooks=hooks,
                                           checkpoint_dir=model_path, save_checkpoint_secs=None,
                                           save_summaries_secs=None) as mon_sess:
        saver.restore(mon_sess, tf.train.latest_checkpoint(model_path))
        step = 0
        preds = []
        labels = []
        wf_result = open(output_file, "w")
        try:
            while not mon_sess.should_stop():
                step += 1
                format_data_infos = mon_sess.run([inputs_task])
                data_dicts = create_or_update_placeholder(format_data_infos[0], None, test_model, configs["task"],
                                                          configs["num_edge_types"], "test")

                preds_subs = list(mon_sess.run(test_model.output, feed_dict=data_dicts)[0])

                labels_subs = list(format_data_infos[0][-1])

                preds.append(preds_subs)
                labels.append(labels_subs)

                metrics_values_names = calculate_evulations(configs["task"], np.concatenate(preds, axis=0),
                                                            np.concatenate(labels, axis=0), configs["nclass"])

                store_lines = ["{}:{}".format(str(metrics_values_names[0][i]), str(metrics_values_names[1][i])) for i in
                               range(len(metrics_values_names[0]))]

                # print(store_lines)
                wf_result.write("\t".join(store_lines) + "\n")
        finally:
            wf_result.close()


def main(argv=None):
    if FLAGS.dataset == "aminer":
        configs = aminer_configures
    elif FLAGS.dataset == "acm":
        configs = acm_configures
    elif FLAGS.dataset == "imdb":
        configs = imdb_configures
    elif FLAGS.dataset == "dblp":
        configs = dblp_configures
    elif FLAGS.dataset == "alibaba":
        configs = alibaba_configures
    elif FLAGS.dataset == 'yelp':
        configs = yelp_configures
    else:
        configs = None
        exit()
    basic_parameters = [FLAGS.dataset, FLAGS.sample_method, FLAGS.sample_type, FLAGS.self_normalize, str(FLAGS.epochs),
                        str(FLAGS.batch_size), str(FLAGS.num_sample), str(FLAGS.seed), str(FLAGS.xi)]
    path = generate_specific_model_path(FLAGS.model_path, basic_parameters)

    loader = HIG(configs)

    model = HeteSamp
    if FLAGS.is_train == "train":
        if FLAGS.sample_method == "adaptive":
            train_fast(path, model, configs, loader)
        else:
            train_fast(path, model, configs, loader)
    elif FLAGS.is_train == "valid":
        output_file = path + "valid_result.txt"
        test(path, model, configs["valid_name"], output_file, configs, loader)
    else:
        output_file = path + "test_result.txt"
        test(path, model, configs["test_name"], output_file, configs, loader)


if __name__ == '__main__':
    tf.app.run()
