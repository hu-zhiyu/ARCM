import math
import numpy as np
import glob
import json
import os
import time
import copy
import sys
import open3d as o3d
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOM_DIR = os.path.join(BASE_DIR + '/log5_test/test_results') 
sys.path.append(ROOM_DIR)

print 'BASE_DIR:', BASE_DIR 
print 'ROOM_DIR:', ROOM_DIR 
savepath = BASE_DIR + '/log5_test/saved_imgs'
if not os.path.exists(savepath): os.mkdir(savepath)
room_name_list_pred = [room_name.strip('\n') for room_name in open(os.path.join(BASE_DIR, 'output_filelist.txt'))]
room_name_list_gt = [room_name[:-9]+ '_gt.txt' for room_name in room_name_list_pred]
total_room_number = len(room_name_list_pred)
print 'Number of rooms in this area:', total_room_number
for files_to_delete in glob.glob(os.path.join(ROOM_DIR, 'cropped*')):
	os.remove(files_to_delete)

label2color = { 0: [0,255,0], # ceiling, green
                1: [255,0,0], # floor, red
                2: [0,255,255], # wall, cyan
                3: [200,200,200], # beam, gray
                4: [255,0,255], # column, magenta(kind of purple)
                5: [100,100,255], # window, royalblue(kind of purple)
                6: [200,200,100], # door, burlywood
                7: [170,120,200], # table, lightpurple
                8: [255,192,203], # chair, pink
                9: [200,100,100], # sofa, indianred
                10: [10,200,100], # bookcase, limegreen
                11: [255,147,0], # board, orange
                12: [255,255,0] # clutter, yellow
                }


def label_to_color(pc=None):
    assert pc.ndim == 1
    sem_color = np.empty([pc.shape[0],3], dtype=int)
    for i in range(pc.shape[0]):
        sem_color[i] = label2color[pc[i]]
    return sem_color

def point_label_to_obj(input_filename1, out_filename):
    data = np.loadtxt(input_filename1)
    coordinates = data[:, 0:3]  # 1930000*3
    sem_label = data[:, -2].astype(int)  # 1930000
    ins_label = data[:, -1].astype(int)  # 1930000
    fout = open(out_filename, 'w')
    for i in range(data.shape[0]):
        sem_color = sem_ins_label2color[sem_label[i]]
        ins_color = sem_ins_label2color[ins_label[i]]
        fout.write('%f %f %f %d %d %d %d %d %d\n' %
                       (coordinates[i, 0], coordinates[i, 1], coordinates[i, 2],
                        sem_color[0], sem_color[1], sem_color[2],
                        ins_color[0], ins_color[1], ins_color[2]))
    fout.close()

def draw_geometry_with_saveimg_callback(pcd, path, window_name='Open3D'):
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imsave(path, np.asarray(image))
        return False
    o3d.visualization.draw_geometries_with_animation_callback([pcd], capture_image, window_name)

def custom_draw_geometry_with_key_callback(pcd, path, window_name='Open3D'):
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imsave(path, np.asarray(image))
        return False
    key_to_callback = {}
    key_to_callback[ord("S")] = capture_image
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback, window_name)

def draw_pointclouds(pc=None, gt_label=None, room_name=None):
    if pc is None or gt_label is None:
        print("No input file is given!")
        exit
    pcd_input = o3d.PointCloud()  # input point cloud (XYZ + RGB)
    pcd_pred = o3d.PointCloud()  # prediction point cloud (XYZ + RGB)
    pcd_gt = o3d.PointCloud()  # ground truth point cloud (XYZ + RGB)

    pcd_input.points = o3d.Vector3dVector(pc[:, 0:3]) # XYZ coordinates
    pcd_pred.points = o3d.Vector3dVector(pc[:, 0:3]) # XYZ coordinates
    pcd_gt.points = o3d.Vector3dVector(pc[:, 0:3]) # XYZ coordinates
    print 'Number of points in this area:', gt_label.shape[0]
    pcd_input.colors = o3d.Vector3dVector(pc[:, 3:6]/ 255.0)  # open3d requires colors (RGB) to be in range[0,1]
    pcd_pred.colors = o3d.Vector3dVector(label_to_color(pc[:, -1])/ 255.0)
    pcd_gt.colors = o3d.Vector3dVector(label_to_color(gt_label)/ 255.0)

    # render and crop the input point cloud
    o3d.visualization.draw_geometries_with_editing([pcd_input], window_name=room_name)

    # record the cropping operation and params
    times_of_cropping = 0
    for item in os.listdir(BASE_DIR): 
        if item.startswith('cropped') and item.endswith('.json'): 
            times_of_cropping += 1
    print 'times of cropping:',times_of_cropping

    # crop the input, pred and gt point cloud
    pcd_input_cropped = pcd_input
    pcd_pred_cropped = pcd_pred
    pcd_gt_cropped = pcd_gt
    for crop_seq in range(times_of_cropping):     
        spv = o3d.visualization.read_selection_polygon_volume("cropped_%d.json"%(crop_seq+1))
        pcd_input_cropped = spv.crop_point_cloud(pcd_input_cropped)
        pcd_pred_cropped = spv.crop_point_cloud(pcd_pred_cropped)
        pcd_gt_cropped = spv.crop_point_cloud(pcd_gt_cropped)
    
    custom_draw_geometry_with_key_callback(pcd_input_cropped, os.path.join(savepath, room_name+'_1input.png'), 
            window_name = room_name + ' input point cloud')
    custom_draw_geometry_with_key_callback(pcd_pred_cropped, os.path.join(savepath, room_name+'_2pred.png'),
            window_name = room_name + ' prediction result')
    custom_draw_geometry_with_key_callback(pcd_gt_cropped, os.path.join(savepath, room_name+'_3gt.png'),
            window_name = room_name + ' ground truth')
    # o3d.visualization.draw_geometries_with_editing([pcd_pred_cropped], window_name = room_name + ' prediction result')
    # o3d.visualization.draw_geometries_with_editing([pcd_gt_cropped], window_name = room_name + ' ground truth')
    # o3d.visualization.draw_geometries_with_editing([pcd])
    # filename = 'test' + '_input_pc.png'
    # path = os.path.join(savepath, filename)
    # draw_geometry_with_saveimg_callback(pcd, path)

def select_points():
    room_name_list_pred = [room_name.strip('\n') for room_name in open(os.path.join(BASE_DIR, 'output_filelist.txt'))]
    room_name_list_gt = [room_name[:-9]+ '_gt.txt' for room_name in room_name_list_pred]
    category_wise_pts_num = np.zeros((13), dtype=np.int32)
    category_wise_pts_freq = np.zeros((13), dtype=np.float64)
    len_room_name_list = len(room_name_list_gt)
    total_num_points = 0
    total_selected_points = 0
    for i in range(len_room_name_list):
        room_name_gt = room_name_list_gt[i]
        gt_label = np.loadtxt(room_name_gt)
        total_num_points += gt_label.shape[0]
        for j in range(category_wise_pts_num.shape[0]):
            category_wise_pts_num[j] += np.size(np.argwhere(gt_label==j))
    print 'Number of points in total:', total_num_points
    for i in range(category_wise_pts_num.shape[0]):
        print 'Number of points in category #%d:'%(i), category_wise_pts_num[i]
        category_wise_pts_freq[i] = 100.0/category_wise_pts_num[i]
    for i in range(len_room_name_list):
        fea_select_file = room_name_list_gt[i][:-7] + '_select.txt'
        fout_fea_select = open(fea_select_file, 'w')
        room_name_gt = room_name_list_gt[i]
        gt_label = np.loadtxt(room_name_gt)
        for j in range(gt_label.shape[0]):
            idx = int(gt_label[j])
            temp = np.random.uniform() < category_wise_pts_freq[idx]
            total_selected_points += temp
            fout_fea_select.write('%d\n'%(temp))
        fout_fea_select.close()
    print 'Number of selected points in total (expecting somewhere near 1300):', total_selected_points


if __name__ == "__main__":
    for i in range(total_room_number):
        room_name_pred = room_name_list_pred[i]
        room_name_gt = room_name_list_gt[i]
        room_name = room_name_pred.split('/')[-1][:-9]
        print '-----------------**************-----------------'
        print 'Processing #%d in #%d'%(i, total_room_number), room_name
        pc = np.loadtxt(room_name_pred)
        gt_label = np.loadtxt(room_name_gt)
        draw_pointclouds(pc, gt_label, room_name)
        for files_to_delete in glob.glob(os.path.join(BASE_DIR, 'cropped*')):
	        os.remove(files_to_delete)
