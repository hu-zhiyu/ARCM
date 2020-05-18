import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
from scipy import stats
import provider
from model import *
from test_utils import *
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import indoor3d_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--verbose', action='store_true', help='if specified, output color-coded seg obj files')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--bandwidth', type=float, default=1., help='Bandwidth for meanshift clustering [default: 1.]')
parser.add_argument('--input_list', type=str, default='data/test_hdf5_file_list_Area5.txt', help='Input data list file')
parser.add_argument('--model_path', type=str, default='log/model.ckpt', help='Path of model')
FLAGS = parser.parse_args()


BATCH_SIZE = 1
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu
MODEL_PATH = FLAGS.model_path
TEST_FILE_LIST = FLAGS.input_list
BANDWIDTH = FLAGS.bandwidth


output_verbose = FLAGS.verbose 

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

OUTPUT_DIR = os.path.join(LOG_DIR, 'test_results')
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

os.system('cp inference_merge.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_inference.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 4096
NUM_CLASSES = 13
NEW_NUM_CLASSES = 13

HOSTNAME = socket.gethostname()
print("ROOTDIR", ROOT_DIR)
ROOM_PATH_LIST = [os.path.join(ROOT_DIR,line.rstrip()) for line in open(os.path.join(ROOT_DIR, FLAGS.input_list))]
len_pts_files = len(ROOM_PATH_LIST)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def test():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, sem_labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Get model and loss
            pred_sem, end_points = get_model(pointclouds_pl, NUM_CLASSES, is_training_pl, extra_constraint=False)
            pred_sem_softmax = tf.nn.softmax(pred_sem)
            pred_sem_label = tf.argmax(pred_sem_softmax, axis=2)

            loss = get_loss(pred_sem, sem_labels_pl, end_points, False)


            loader = tf.train.Saver()


        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        is_training = False

        # Restore variables from disk.
        loader.restore(sess, MODEL_PATH)
        log_string("Model restored.")

        ops = {'pointclouds_pl': pointclouds_pl,
               'sem_labels_pl': sem_labels_pl,
               'is_training_pl': is_training_pl,
               'pred_sem_label': pred_sem_label,
               'pred_sem_softmax': pred_sem_softmax,
               'loss': loss}

        total_acc = 0.0
        total_seen = 0

        output_filelist_f = os.path.join(LOG_DIR, 'output_filelist.txt')
        fout_out_filelist = open(output_filelist_f, 'w')
        for shape_idx in range(len_pts_files):
            room_path = ROOM_PATH_LIST[shape_idx]
            log_string('%d / %d ...' % (shape_idx, len_pts_files))
            log_string('Loading train file ' + room_path)
            out_data_label_filename = os.path.basename(room_path)[:-4] + '_pred.txt'
            out_data_label_filename = os.path.join(OUTPUT_DIR, out_data_label_filename)
            out_gt_label_filename = os.path.basename(room_path)[:-4] + '_gt.txt'
            out_gt_label_filename = os.path.join(OUTPUT_DIR, out_gt_label_filename)
            fout_data_label = open(out_data_label_filename, 'w')
            fout_gt_label = open(out_gt_label_filename, 'w')

            fout_out_filelist.write(out_data_label_filename+'\n')

            cur_data, cur_sem, _ = indoor3d_util.room2blocks_wrapper_normalized(room_path, NUM_POINT, block_size=1.0, stride=0.9,
                                                 random_sample=False, sample_num=None)
            cur_data = cur_data[:, 0:NUM_POINT, :]
            cur_sem = np.squeeze(cur_sem)
            # Get room dimension..
            data_label = np.load(room_path)
            data = data_label[:, 0:6]
            max_room_x = max(data[:, 0])
            max_room_y = max(data[:, 1])
            max_room_z = max(data[:, 2])

            cur_pred_sem = np.zeros_like(cur_sem)
            cur_pred_sem_softmax = np.zeros([cur_sem.shape[0], cur_sem.shape[1], NUM_CLASSES])

            num_data = cur_data.shape[0]
            for j in range(num_data):
                log_string("Processsing: Shape [%d] Block[%d]"%(shape_idx, j))

                pts = cur_data[j,...]
                sem = cur_sem[j]

                feed_dict = {ops['pointclouds_pl']: np.expand_dims(pts, 0),
                             ops['sem_labels_pl']: np.expand_dims(sem, 0),
                             ops['is_training_pl']: is_training}

                _, pred_sem_label_val, pred_sem_softmax_val = sess.run(
                    [ops['loss'], ops['pred_sem_label'], ops['pred_sem_softmax']],
                    feed_dict=feed_dict)

                pred_sem = np.squeeze(pred_sem_label_val, axis=0)
                pred_sem_softmax = np.squeeze(pred_sem_softmax_val, axis=0)
                cur_pred_sem[j, :] = pred_sem
                cur_pred_sem_softmax[j, ...] = pred_sem_softmax
                total_acc += float(np.sum(pred_sem==sem))/pred_sem.shape[0]
                total_seen += 1

            seg_pred = cur_pred_sem.reshape(-1)
            seg_pred_softmax = cur_pred_sem_softmax.reshape([-1, NUM_CLASSES])
            pts = cur_data.reshape([-1, 9])            
            seg_gt = cur_sem.reshape(-1)

            if output_verbose:
                pts[:, 6] *= max_room_x
                pts[:, 7] *= max_room_y
                pts[:, 8] *= max_room_z
                pts[:, 3:6] *= 255.0
                sem = seg_pred.astype(np.int32)
                sem_softmax = seg_pred_softmax
                sem_gt = seg_gt
                for i in range(pts.shape[0]):
                    fout_data_label.write('%f %f %f %d %d %d %f %d\n' % (
                    pts[i, 6], pts[i, 7], pts[i, 8], pts[i, 3], pts[i, 4], pts[i, 5], sem_softmax[i, sem[i]], sem[i]))
                    fout_gt_label.write('%d\n' % (sem_gt[i]))

            fout_data_label.close()
            fout_gt_label.close()


        fout_out_filelist.close()

if __name__ == "__main__":
    test()
    LOG_FOUT.close()
