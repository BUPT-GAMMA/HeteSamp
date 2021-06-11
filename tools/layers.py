import tensorflow as tf


def dense_layers(input_tensor, out_dim, name, keep_prob=1.0, norm_rate=0.0, activation=None, bias = False):
    regularizer = tf.contrib.layers.l2_regularizer(norm_rate)

    outs = tf.layers.dense(input_tensor, out_dim, activation=activation, kernel_regularizer=regularizer,
                           reuse=tf.AUTO_REUSE, use_bias=bias, name=name)

    if 0 < keep_prob < 1.0:
        outs = tf.layers.dropout(outs, rate=1 - keep_prob)

    return outs
