# TRACK STATISTICS GROUND TRUTH

import numpy as np
import h5py
import sys
import os
import itertools
import networkx as nx
# import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
# import matplotlib.animation as animation

INTERVAL = 20
SKIP = 0
VOXEL_RESOLUTION = 0.3
DOWNSAMPLE_RESOLUTION = 0.3 #0.1
MIN_CLUSTER = 50

sequence='07'

class Track:
    def __init__(self, obj_id):
        self.obj_id = obj_id
        self.centroids = []
        self.frames = []
        self.pointcount = []
        self.length = 0
        self.moving = False
        self.motion = []
        self.averagemotion = 0
        
    def Extend(self,centroid,frame,pointcount):
        self.centroids.append(centroid)
        self.frames.append(frame)
        self.pointcount.append(pointcount)
        self.length += 1

def downsample(cloud, resolution):
    voxel_coordinates = [tuple(p) for p in np.round((cloud[:,:3] / resolution)).astype(int)]
    voxel_set = set()
    downsampled_cloud = []
    for i in range(len(cloud)):
        if not voxel_coordinates[i] in voxel_set:
            voxel_set.add(voxel_coordinates[i])
            downsampled_cloud.append(cloud[i])
    return np.array(downsampled_cloud)

def sample_sphere(radius,trans,npts):
    sphere = []
    azis=(np.random.random(npts)-0.5)*2*np.pi
    eles=(np.random.random(npts)-0.5)*np.pi
    for i in range(npts):
        pt=[radius*np.sin(eles[i])*np.cos(azis[i]),radius*np.sin(eles[i])*np.sin(azis[i]),radius*np.cos(eles[i])]
        for j in range(3):
            pt[j]+=trans[j]
        sphere.append(pt)
    return sphere

def sample_cube(side,trans,npts):
    cube = []
    us=np.random.random(npts)-0.5
    vs=np.random.random(npts)-0.5
    faces = np.random.randint(0,6,npts)
    
    rot1=np.random.random()*360
    rot2=np.random.random()*360
    rot3=np.random.random()*360
    r = R.from_euler('zyx', [rot1, rot2, rot3], degrees=True)
    
    for i in range(npts):
        if faces[i]==0:
            pt=[ 0.5*side,us[i]*side,vs[i]*side]
        if faces[i]==1:
            pt=[-0.5*side,us[i]*side,vs[i]*side]
        if faces[i]==2:
            pt=[us[i]*side, 0.5*side,vs[i]*side]
        if faces[i]==3:
            pt=[us[i]*side,-0.5*side,vs[i]*side]
        if faces[i]==4:
            pt=[us[i]*side,vs[i]*side, 0.5*side]
        if faces[i]==5:
            pt=[us[i]*side,vs[i]*side,-0.5*side]
        ptr=r.apply(pt)
        for j in range(3):
            pt[j]+=trans[j]
        cube.append(pt)
    return cube
    

# get camera calibration
calib_file = open('Z:\data_odometry_velodyne\dataset\sequences\\' + sequence + '\calib.txt', 'r')
calib = {}
for line in calib_file:
    key, content = line.strip().split(":")
    values = [float(v) for v in content.strip().split()]
    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0
    calib[key] = pose
calib_file.close()

# get poses
poses = []
Tr = calib["Tr"]
Tr_inv = np.linalg.inv(Tr)
pose_file = open('Z:\data_odometry_velodyne\dataset\sequences\\' + sequence + '\poses.txt', 'r')
for line in pose_file:
    values = [float(v) for v in line.strip().split()]
    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0
    poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
pose_file.close()

scan_paths = 'Z:\data_odometry_velodyne\dataset\sequences\\' + sequence + '\\velodyne'
scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_paths)) for f in fn]
# scan_names=os.listdir(scan_paths)
scan_names.sort()
print('Scans: ',len(scan_names))
    
label_paths = 'Z:\data_odometry_velodyne\dataset\sequences\\' + sequence + '\labels'
label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_paths)) for f in fn]
label_names.sort()
print('Labels: ',len(label_names))

stacked_points = []
counts = []

frames = len(scan_names)
offset = INTERVAL-1
tracks = []

all_points = []

obj_ids_uq = []
obj_ids_tr = []



for k in range(len(scan_names)):
    if k%50 == 0:
        print(k,' of ',len(scan_names))
    scan = np.fromfile(scan_names[k], dtype=np.float32)
    scan = scan.reshape((-1, 4))
    
    # get XYZ in world coordinate frame
    xyz_local = scan[:, 0:3]
    R_world_local = poses[k][:3, :3]
    t_world_local = poses[k][:3, 3]
    xyz_world = xyz_local.dot(R_world_local.T) + t_world_local
#     xyz_voxels = [tuple(v) for v in np.round(xyz_world / VOXEL_RESOLUTION).astype(int)]
    
    # get point labels
    label = np.fromfile(label_names[k], dtype=np.uint32)
    obj_id = [l >> 16 for l in label]
    cls_id = [l & 0xFFFF for l in label]
    
    # stack in Nx5 array
    points = np.zeros((len(xyz_world), 6))
    points[:, :3] = xyz_world
    points[:, 3] = obj_id
    points[:, 4] = cls_id
    
    points_sorted=points[points[:, 3].argsort()]
    obj_id_sorted=points_sorted[:, 3]
    unique_obj_ids=np.unique(obj_id_sorted)
    point_count=0
    
    for obj_id_unique in unique_obj_ids:
        track_index=-1
        for j in range(len(tracks)):
            if(tracks[j].obj_id==obj_id_unique):
                track_index = j
                break
        if track_index == -1:
            track_index = len(tracks)
            tracks.append(Track(obj_id_unique))
            if not(obj_id_unique in obj_ids_uq):
                obj_ids_uq.append(obj_id_unique)
                obj_ids_tr.append(track_index)

        obj_pts=points_sorted[(obj_id_sorted==obj_id_unique).astype(bool),:]
        point_count+=obj_pts.shape[0]
        obj_cent=np.mean(obj_pts, axis=0)
        tracks[track_index].Extend(obj_cent[0:3],k,obj_pts.shape[0])
        
N_TRACKS=len(tracks)

for i in range(N_TRACKS):
    if tracks[i].length >1:
        for j in range(1,tracks[i].length):
            delta = tracks[i].centroids[j]-tracks[i].centroids[j-1]
            tracks[i].motion.append(delta)
            tracks[i].averagemotion+=np.linalg.norm(delta)
        tracks[i].averagemotion/=float(tracks[i].length)
        if tracks[i].averagemotion > 2.5:
            tracks[i].moving = True
        
print(N_TRACKS, ' Unique Tracks found')
for i in range(N_TRACKS):
    if tracks[i].length > 1:
        print('Track: ',i,' with length: ',tracks[i].length,' average speed: ',tracks[i].averagemotion,
              ' average points: ',sum(tracks[i].pointcount)/float(tracks[i].length))
    
        
for u in range(len(obj_ids_uq)):
    print('Object ID: ',obj_ids_uq[u],' is associated with Track: ',obj_ids_tr[u])

file=open('Z:/PHD/pcseg/TRACKS/KITTI_'+sequence+'.csv','w')
for tr in tracks:
    for j in range(tr.length):
        strfil ='%d,%d,%.4f,%.4f,%.4f,%d,%d\n'%(tr.obj_id,j,tr.centroids[j][0],tr.centroids[j][1],tr.centroids[j][2],tr.frames[j],tr.pointcount[j])
        file.write(strfil)
file.close()