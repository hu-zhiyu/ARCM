import tensorflow as tf

def calc_sim_map(l1_label, l2_label):
    batch_size = l1_label.shape[0]
    assert l1_label.shape[0] == l2_label.shape[0]
    num_points_l1 = l1_label.shape[-1]
    num_points_l2 = l2_label.shape[-1]
    l1_label_expanded = tf.tile(tf.reshape(l1_label,[batch_size, num_points_l1, 1]), [1, 1, num_points_l2])
    l2_label_expanded = tf.tile(tf.reshape(l2_label,[batch_size, 1, num_points_l2]), [1, num_points_l1, 1])
    pair_label = tf.cast(tf.equal(l1_label_expanded, l2_label_expanded), dtype=tf.float32)
    pair_label_row_sum = tf.tile(tf.expand_dims(tf.reduce_sum(pair_label, axis=2),axis=2), [1,1,num_points_l2])
    sim_map = pair_label / pair_label_row_sum
    # print('sim_map', sim_map)
    sim_map = pair_label/pair_label_row_sum
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     sim_map = sess.run(sim_map)
    #     print(sum(sim_map[0,0,:]))
    return sim_map

if __name__ == '__main__':
    l1_label = tf.random_uniform((2,1024))
    l2_label = tf.random_uniform((2,256))
    l1_label = tf.where(l1_label<0.5, x=tf.zeros_like(l1_label), y=tf.ones_like(l1_label))
    l2_label = tf.where(l2_label<0.5, x=tf.zeros_like(l2_label), y=tf.ones_like(l2_label))
    print('l1_label:', l1_label.shape)
    calc_sim_map(l1_label, l2_label)
