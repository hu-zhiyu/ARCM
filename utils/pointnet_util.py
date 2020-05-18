""" PointNet++ Layers
Author: Charles R. Qi
Date: November 2017
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util

def sample_and_group(npoint, radius, nsample, xyz, points, global_idx, knn=False, use_xyz=True):
    fps_idx, g_idx = farthest_point_sample(npoint, xyz, global_idx)
    new_xyz = gather_point(xyz, fps_idx) # (batch_size, npoint, 3)
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    center_xyz = tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])
    diff_xyz = grouped_xyz - center_xyz # translation normalization
    euclid_dist = tf.norm(diff_xyz, axis=-1, keep_dims=True)
    grouped_xyz = tf.concat([grouped_xyz, center_xyz, diff_xyz, euclid_dist], axis=-1)
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([diff_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = diff_xyz
    return new_xyz, new_points, fps_idx, g_idx, diff_xyz


def pointnet_sa_module(xyz, global_idx, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, g_idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, global_idx, knn, use_xyz)

        # Point Feature Embedding
        xyz_coordinates = grouped_xyz

        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            pass
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)
        elif pooling=='attentive_pooling':
            points_fea = new_points
            xyz_fea_1 = tf_util.conv2d(xyz_coordinates, mlp2[0], [1, 1],
                                       padding='VALID', stride=[1, 1],
                                       bn=bn, is_training=is_training,
                                       scope='coor_mlp1', bn_decay=bn_decay)
            xyz_fea_2 = tf_util.conv2d(xyz_fea_1, mlp2[1], [1, 1],
                                       padding='VALID', stride=[1, 1],
                                       bn=bn, is_training=is_training,
                                       scope='coor_mlp2', bn_decay=bn_decay)
            xyz_fea_3 = tf_util.conv2d(xyz_fea_2, mlp2[2], [1, 1],
                                       padding='VALID', stride=[1, 1],
                                       bn=bn, is_training=is_training,
                                       scope='coor_mlp3', bn_decay=bn_decay)

            xyz_fea = tf.concat([xyz_fea_1, xyz_fea_2, xyz_fea_3], axis=-1)
            fea = tf.concat([xyz_fea, points_fea], axis=3)

            fea = tf_util.conv2d(fea, points_fea.shape[-1], [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=bn, is_training=is_training,
                                 scope='final_coor_mlp_1', bn_decay=bn_decay)
            fea = tf.transpose(fea, [0, 1, 3, 2])  # (batchsize, npoints, fea_dim, nsample)
            att_weights1 = tf.nn.softmax(fea)
            fea1 = tf.multiply(fea, att_weights1)

            fea2 = fea1 + fea
            fea2 = tf.transpose(fea2, [0, 1, 3, 2])
            fea2 = tf_util.conv2d(fea2, points_fea.shape[-1], [1, 1],
                                  padding='VALID', stride=[1, 1],
                                  bn=bn, is_training=is_training,
                                  scope='final_coor_mlp_2', bn_decay=bn_decay)
            fea2 = tf.transpose(fea2, [0, 1, 3, 2])  # (batchsize, npoints, fea_dim, nsample)
            att_weights2 = tf.nn.softmax(fea2)
            fea2 = tf.multiply(fea2, att_weights2)
            new_points = tf.reduce_sum(fea2, axis=[-1], keep_dims=True, name='sum_pool')
            new_points = tf.transpose(new_points, [0,1,3,2])
        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx, g_idx

 
def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)
        if points1 is not None:
            # gated fusion
            concated_fea = tf.concat(axis=2, values=[interpolated_points, points1]) # B, ndataset1, nchannel1 + nchannel2
            concated_fea = tf.nn.relu(concated_fea)
            concated_fea = tf.expand_dims(concated_fea, 2)
            concated_fea = tf_util.conv2d(concated_fea, 1, [1, 1],
                                         padding='VALID', stride=[1, 1],
                                         bn=bn, is_training=is_training,
                                         scope='concat_conv', bn_decay=bn_decay)
            concated_fea = tf.squeeze(concated_fea, 2)
            fusion_weights = tf.sigmoid(concated_fea)
            interp_fusion_weights = tf.ones_like(fusion_weights) - fusion_weights
            new_points1 = tf.concat(axis=2, values=[tf.multiply(interpolated_points, interp_fusion_weights), tf.multiply(points1, fusion_weights)])
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1


def non_local_block(points, is_training, bn_decay, scope, bn=True, use_nchw=False):
    with tf.variable_scope(scope) as sc:
        output_channel = points.shape[-1]
        newpoints = tf.expand_dims(points, 2)
        query = tf_util.conv2d(newpoints, output_channel, [1, 1],
                                         padding='VALID', stride=[1, 1],
                                         bn=bn, is_training=is_training,
                                         scope='query', bn_decay=bn_decay)
        query = tf.squeeze(query, [2])

        key = tf_util.conv2d(newpoints, output_channel, [1, 1],
                                         padding='VALID', stride=[1, 1],
                                         bn=bn, is_training=is_training,
                                         scope='key', bn_decay=bn_decay)
        key = tf.squeeze(key, [2])
        key = tf.transpose(key, [0,2,1])

        value = tf_util.conv2d(newpoints, output_channel, [1, 1],
                                         padding='VALID', stride=[1, 1],
                                         bn=bn, is_training=is_training,
                                         scope='value', bn_decay=bn_decay)
        value = tf.squeeze(value, [2])
        # spatial attention
        spatial_affinity = tf.matmul(query, key)
        spatial_affinity = (int(output_channel)**-.5) * spatial_affinity
        spatial_weights = tf.nn.softmax(spatial_affinity)
        spatial_attention_feature = tf.matmul(spatial_weights, value)
        spatial_attention_feature = spatial_attention_feature + value
        return spatial_attention_feature


if __name__ == '__main__':
    xyz1 = tf.random_normal([2,64,3])
    xyz2 = tf.random_normal([2,4,3])
    points1 = tf.random_normal([2,64,16])
    points2 = tf.random_normal([2,4,32])
    mlp = [16,32,64]
    fea = pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training=tf.constant(True), bn_decay=None, scope='test', bn=True)
