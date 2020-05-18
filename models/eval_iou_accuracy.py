import os
import numpy as np
from scipy import stats

NUM_CLASSES = 13

pred_data_label_filenames = []
for i in range(5,6):
  file_name = 'log{}_test/output_filelist.txt'.format(i)
  pred_data_label_filenames += [line.rstrip() for line in open(file_name)]

gt_label_filenames = [f.rstrip('_pred\.txt') + '_gt.txt' for f in pred_data_label_filenames]

num_room = len(gt_label_filenames)

# Initialize...
# acc and macc
total_true = 0
total_seen = 0
true_positive_classes = np.zeros(NUM_CLASSES)
positive_classes = np.zeros(NUM_CLASSES)
gt_classes = np.zeros(NUM_CLASSES)
# mIoU
ious = np.zeros(NUM_CLASSES)
totalnums = np.zeros(NUM_CLASSES)
# precision & recall
total_gt_ins = np.zeros(NUM_CLASSES)
at = 0.5
tpsins = [[] for itmp in range(NUM_CLASSES)]
fpsins = [[] for itmp in range(NUM_CLASSES)]
# mucov and mwcov
all_mean_cov = [[] for itmp in range(NUM_CLASSES)]
all_mean_weighted_cov = [[] for itmp in range(NUM_CLASSES)]


for i in range(num_room):
    print(i)
    data_label = np.loadtxt(pred_data_label_filenames[i])
    pred_sem = data_label[:, -1].reshape(-1).astype(np.int)
    gt_label = np.loadtxt(gt_label_filenames[i])
    gt_sem = gt_label.reshape(-1).astype(np.int)
    print(gt_label.shape)

    # semantic acc
    total_true += np.sum(pred_sem == gt_sem)
    total_seen += pred_sem.shape[0]

    # pn semantic mIoU
    for j in xrange(gt_sem.shape[0]):
        gt_l = int(gt_sem[j])
        pred_l = int(pred_sem[j])
        gt_classes[gt_l] += 1
        positive_classes[pred_l] += 1
        true_positive_classes[gt_l] += int(gt_l==pred_l)

# semantic results
iou_list = []
for i in range(NUM_CLASSES):
    iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i]) 
    print(iou)
    iou_list.append(iou)

log_string('Semantic Segmentation oAcc: {}'.format(sum(true_positive_classes)/float(sum(positive_classes))))
#log_string('Semantic Segmentation Acc: {}'.format(true_positive_classes / gt_classes))
log_string('Semantic Segmentation mAcc: {}'.format(np.mean(true_positive_classes / gt_classes)))
log_string('Semantic Segmentation IoU: {}'.format(iou_list))
log_string('Semantic Segmentation mIoU: {}'.format(1.*sum(iou_list)/NUM_CLASSES))
