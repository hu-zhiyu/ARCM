import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tensorflow as tf
import numpy as np
import tf_util
import gather_util
from sim_map_util import calc_sim_map
from pointnet_util import pointnet_sa_module, pointnet_fp_module, non_local_block
print ROOT_DIR

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 9))
    sem_labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, sem_labels_pl

def get_model(point_cloud, num_class, is_training, extra_constraint=True, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud[:,:,:3]
    l0_points = point_cloud[:,:,3:]
    # end_points['l0_xyz'] = l0_xyz
    global_idx_init = tf.zeros(shape=(batch_size, num_point), dtype=tf.int32)

    # Layer 1
    l1_xyz, l1_points, l1_indices, l1_g_idx = pointnet_sa_module(l0_xyz, global_idx_init, l0_points, npoint=1024, radius=0.1, nsample=32, mlp=[32,32,64], mlp2=[64,64,64], group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1', pooling='attentive_pooling')
    l2_xyz, l2_points, l2_indices, l2_g_idx = pointnet_sa_module(l1_xyz, l1_indices, l1_points, npoint=256, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=[64,64,64], group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2', pooling='attentive_pooling')
    l3_xyz, l3_points, l3_indices, l3_g_idx = pointnet_sa_module(l2_xyz, l2_g_idx, l2_points, npoint=64, radius=0.4, nsample=32, mlp=[128,128,256], mlp2=[64,64,64], group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3', pooling='attentive_pooling')
    l4_xyz, l4_points, l4_indices, l4_g_idx = pointnet_sa_module(l3_xyz, l3_g_idx, l3_points, npoint=16, radius=0.8, nsample=32, mlp=[256,256,512], mlp2=[64,64,64], group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4', pooling='attentive_pooling')
    # Feature Propagation layers
    l1_g_idx = l1_indices
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='fp1', bn=True)
    l3_points = non_local_block(l3_points, is_training, bn_decay, scope="non_local1", bn=True, use_nchw=False)
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fp2', bn=True)
    l2_points = non_local_block(l2_points, is_training, bn_decay, scope="non_local2", bn=True, use_nchw=False)
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fp3', bn=True)
    l1_points = non_local_block(l1_points, is_training, bn_decay, scope="non_local3", bn=True, use_nchw=False)
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='fp4',bn=True)

    # FC layers
    end_points.update({'l0_points':l0_points})
    end_points.update({'l1_g_idx': l1_g_idx, 'l2_g_idx': l2_g_idx, 'l3_g_idx': l3_g_idx, 'l4_g_idx': l4_g_idx})
    l0_net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc01', bn_decay=bn_decay)
    l0_net = tf_util.dropout(l0_net, keep_prob=0.5, is_training=is_training, scope='dp01')
    l0_net = tf_util.conv1d(l0_net, num_class, 1, padding='VALID', activation_fn=None, scope='fc02')

    if extra_constraint:
        l1_net = tf_util.conv1d(l1_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc11', bn_decay=bn_decay)
        l1_net = tf_util.dropout(l1_net, keep_prob=0.5, is_training=is_training, scope='dp11')
        l1_net = tf_util.conv1d(l1_net, num_class, 1, padding='VALID', activation_fn=None, scope='fc12')
        l2_net = tf_util.conv1d(l2_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc21', bn_decay=bn_decay)
        l2_net = tf_util.dropout(l2_net, keep_prob=0.5, is_training=is_training, scope='dp21')
        l2_net = tf_util.conv1d(l2_net, num_class, 1, padding='VALID', activation_fn=None, scope='fc22')    
        l3_net = tf_util.conv1d(l3_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc31', bn_decay=bn_decay)
        l3_net = tf_util.dropout(l3_net, keep_prob=0.5, is_training=is_training, scope='dp31')
        l3_net = tf_util.conv1d(l3_net, num_class, 1, padding='VALID', activation_fn=None, scope='fc32')
        end_points.update({'l1_net':l1_net, 'l2_net':l2_net, 'l3_net':l3_net})

    return l0_net, end_points


def get_loss(pred, label, end_points, is_training):

    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred)
    tf.summary.scalar('classify loss', classify_loss)

    if is_training == False:
        return classify_loss

    l1_label = gather_util.custom_gather(label, end_points['l1_g_idx'])  # (B, 1024)
    l2_label = gather_util.custom_gather(label, end_points['l2_g_idx'])  # (B, 256)
    l3_label = gather_util.custom_gather(label, end_points['l3_g_idx'])  # (B, 64)

    sim_map_loss = tf.losses.sparse_softmax_cross_entropy(labels=l1_label, logits=end_points['l1_net']) + \
        tf.losses.sparse_softmax_cross_entropy(labels=l2_label, logits=end_points['l2_net']) + \
        tf.losses.sparse_softmax_cross_entropy(labels=l3_label, logits=end_points['l3_net'])

    tf.summary.scalar('sim_map loss', sim_map_loss)
    loss = classify_loss + sim_map_loss
    return loss, classify_loss, sim_map_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.random_uniform((2,2048,3))
        label = tf.ones((2, 2048), dtype=tf.int32)
        l0_net, end_points = get_model(inputs, 10, tf.constant(True), True)
        print('l0_net', l0_net)
