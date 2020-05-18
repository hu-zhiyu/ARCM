"""
    Author: Zhiyu Hu (Beihang University)
    Date: March 3, 2020
"""
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
import threading
CATEGORY = 'Lamp'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
SHAPE_DIR = os.path.join(BASE_DIR, 'shapenetpart_vis', CATEGORY) 
sys.path.append(SHAPE_DIR)

print 'BASE_DIR:', BASE_DIR 
print 'SHAPE_DIR:', SHAPE_DIR 
shape_list = []
for file_name in os.listdir(SHAPE_DIR):
    if file_name.endswith('.txt'):
        shape_list.append(file_name)
shape_list = ['470.txt']
shape_num = len(shape_list)
print '%d shapes to be processed'%(shape_num)
# shape_name_list_pred = [shape_name.strip('\n') for shape_name in open(os.path.join(BASE_DIR, 'output_filelist.txt'))]
# shape_name_list_gt = [shape_name[:-9]+ '_gt.txt' for shape_name in shape_name_list_pred]
# total_room_number = len(shape_name_list_pred)
# print 'Number of rooms in this area:', total_room_number
for files_to_delete in glob.glob(os.path.join(SHAPE_DIR, 'cropped*')):
	os.remove(files_to_delete)

label2color = { 0: [0,255,0], 1: [255,0,0], 2: [0,255,255], 3: [255,97,0],
                4: [255,192,203], 5: [10,200,100],
                6: [255,147,0], 7: [0,255,255],
                8: [170,120,200], 9: [255,97,0], 10: [0,255,255], 11: [200,100,100],
                12: [0,255,0], 13: [255,0,0], 14: [255,0,255], 15: [200,200,100],
                16: [200,200,100], 17: [200,100,100], 18: [255,97,0],
                19: [100,100,255], 20: [10,200,100], 21: [255,147,0],
                22: [255,147,0], 23: [0,255,0],
                24: [0,255,0], 25: [255,0,0], 26: [0,255,255], 27: [100,100,255],
                28: [255,147,0], 29: [0,255,255],
                30: [255,192,203], 31: [255,0,0], 32: [0,255,255], 33: [10,200,100],  34: [200,200,100], 35: [100,100,255],
                36: [255,147,0], 37: [0,255,0],
                38: [100,100,255], 39: [10,200,100], 40: [255,147,0],
                41: [200,200,100], 42: [200,100,100], 43: [255,97,0],
                44: [100,100,255], 45: [255,147,0], 46: [200,200,100],
                47: [255,97,0], 48: [255,0,255], 49: [10,200,100],
                50: [30,30,30]
                }


def label_to_color(pc=None):
    assert pc.ndim == 1
    sem_color = np.empty([pc.shape[0],3], dtype=int)
    for i in range(pc.shape[0]):
        sem_color[i] = label2color[pc[i]]
    return sem_color

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

def draw_pointclouds(pc=None, shape_name=None):
    if pc is None:
        print("No input file is given!")
        exit
    pcd_input = o3d.PointCloud()  # input point cloud (XYZ + RGB)
    pcd_pred = o3d.PointCloud()  # prediction point cloud (XYZ + RGB)
    pcd_gt = o3d.PointCloud()  # ground truth point cloud (XYZ + RGB)

    pcd_input.points = o3d.Vector3dVector(pc[:, 0:3]) # XYZ coordinates
    pcd_pred.points = o3d.Vector3dVector(pc[:, 0:3]) # XYZ coordinates
    pcd_gt.points = o3d.Vector3dVector(pc[:, 0:3]) # XYZ coordinates
    # pcd_input.colors = o3d.Vector3dVector([30,30,30]/ 255.0)  # open3d requires colors (RGB) to be in range[0,1]
    pcd_pred.colors = o3d.Vector3dVector(label_to_color(pc[:, -2])/ 255.0)
    pcd_gt.colors = o3d.Vector3dVector(label_to_color(pc[:, -1])/ 255.0)

    # render and crop the input point cloud
    # o3d.visualization.draw_geometries_with_editing([pcd_input], window_name=shape_name)

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
    
    # custom_draw_geometry_with_key_callback(pcd_input_cropped, os.path.join(SHAPE_DIR, shape_name+'_1input.png'), 
    #         window_name = shape_name + ' input point cloud')
    custom_draw_geometry_with_key_callback(pcd_pred_cropped, os.path.join(SHAPE_DIR, shape_name+'_2pred.png'),
            window_name = shape_name + ' prediction result')
    custom_draw_geometry_with_key_callback(pcd_gt_cropped, os.path.join(SHAPE_DIR, shape_name+'_3gt.png'),
            window_name = shape_name + ' ground truth')


if __name__ == "__main__":
    for i in range(shape_num):
        shape_name = shape_list[i]
        print '-----------------**************-----------------'
        print 'Processing %s'%(shape_name)
        path = os.path.join(SHAPE_DIR, shape_name)
        pc = np.loadtxt(path)
        draw_pointclouds(pc, shape_name)
        for files_to_delete in glob.glob(os.path.join(BASE_DIR, 'cropped*')):
	        os.remove(files_to_delete)
