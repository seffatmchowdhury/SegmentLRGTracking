# STAGE SEMANTIC KITTI MODIFIED

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
        
colors=np.random.rand(10000,3)*255

# fig1,ax1 = plt.subplots(figsize=(10, 10))
# rot1=45
# rot2=45
# rot3=0
# r = R.from_euler('zyx', [rot1, rot2, rot3], degrees=True)
# scale = 0.125



# rgb=np.zeros((1000,1000,3),dtype=np.uint8)
# for k in range(N_TRACKS):
#     for j in range(tracks[k].length):
#         vr=r.apply([tracks[k].centroids[j][0],tracks[k].centroids[j][1],tracks[k].centroids[j][2]])
    
#         x=int(vr[0]*scale)+500
#         y=int(vr[1]*scale)+500
#         if (x>=1) and (x<999) and (y>=1) and (y<999):
#             rgb[x,y,0]=int(colors[k,0])
#             rgb[x,y,1]=int(colors[k,1])
#             rgb[x,y,2]=int(colors[k,2])
            
#             rgb[x+1,y,0]=int(colors[k,0])
#             rgb[x+1,y,1]=int(colors[k,1])
#             rgb[x+1,y,2]=int(colors[k,2])
            
#             rgb[x-1,y,0]=int(colors[k,0])
#             rgb[x-1,y,1]=int(colors[k,1])
#             rgb[x-1,y,2]=int(colors[k,2])
            
#             rgb[x,y+1,0]=int(colors[k,0])
#             rgb[x,y+1,1]=int(colors[k,1])
#             rgb[x,y+1,2]=int(colors[k,2])
            
#             rgb[x,y-1,0]=int(colors[k,0])
#             rgb[x,y-1,1]=int(colors[k,1])
#             rgb[x,y-1,2]=int(colors[k,2])
            
# im = ax1.imshow(rgb)
# plt.show()



        
while offset < frames:
    print(offset)
    
    for k in range(offset-INTERVAL+1,offset+1):
        scan = np.fromfile(scan_names[k], dtype=np.float32)
        scan = scan.reshape((-1, 4))

        # get XYZ in world coordinate frame
        xyz_local = scan[:, 0:3]
        R_world_local = poses[k][:3, :3]
        t_world_local = poses[k][:3, 3]
        xyz_world = xyz_local.dot(R_world_local.T) + t_world_local
#         xyz_voxels = [tuple(v) for v in np.round(xyz_world / VOXEL_RESOLUTION).astype(int)]
    
        # get point labels
        label = np.fromfile(label_names[k], dtype=np.uint32)
        obj_id = [l >> 16 for l in label]
        cls_id = [l & 0xFFFF for l in label]
        
        trk_id = []
        for i in obj_id:
            trk_id.append(obj_ids_tr[obj_ids_uq.index(i)])
    
        npts_sphere1 = 300
        npts_sphere2 = 300
        npts_cube1 = 300
        npts_orig = len(xyz_world)
        npts_total = npts_orig + npts_sphere1 + npts_sphere2 + npts_cube1
        # stack in Nx8 array
        points = np.zeros((npts_total, 8))
        points[0:npts_orig, :3] = xyz_world
        points[0:npts_orig, 3] = obj_id
        points[0:npts_orig, 4] = cls_id
        points[0:npts_orig, 5] = k-offset
        points[0:npts_orig, 6] = trk_id
        points[0:npts_orig, 7] = 0
        for i in range(len(xyz_world)):
            if tracks[trk_id[i]].moving:
                points[i, 7] = 1
        
        # Artificial sphere 1
        sphere1_geom = sample_sphere(1,[1.5+(np.sin(float(k)*0.1)*2.0), 5.0+(np.cos(float(k)*0.1)*2.0), 1.0],npts_sphere1)
        npts_start = npts_orig
        for i in range(npts_sphere1):
            points[npts_start+i,0:3]=sphere1_geom[i][0:3] + t_world_local
            points[npts_start+i,3]=5001
            points[npts_start+i,4]=-1
            points[npts_start+i,5]=k-offset
            points[npts_start+i,6]=5001
            points[npts_start+i,7]=1
        
        # Artificial sphere 2
        sphere2_geom = sample_sphere(1.33,[1.5+(np.sin(float(-k)*0.1)*4.0), 5.0+(np.cos(float(-k)*0.1)*4.0), 1.0],npts_sphere2)
        npts_start = npts_orig + npts_sphere1
        for i in range(npts_sphere2):
            points[npts_start+i,0:3]=sphere2_geom[i][0:3] + t_world_local
            points[npts_start+i,3]=5002
            points[npts_start+i,4]=-1
            points[npts_start+i,5]=k-offset
            points[npts_start+i,6]=5002
            points[npts_start+i,7]=1

        # Artificial cube 1
        cube1_geom = sample_cube(1.33,[1.5+(np.sin(float(k)*0.1)*6.0), 5.0+(np.cos(float(k)*0.1)*6.0), 1.0],npts_cube1)
        npts_start = npts_orig + npts_sphere1 + npts_sphere2
        for i in range(npts_cube1):
            points[npts_start+i,0:3]=cube1_geom[i][0:3] + t_world_local
            points[npts_start+i,3]=5003
            points[npts_start+i,4]=-1
            points[npts_start+i,5]=k-offset
            points[npts_start+i,6]=5003
            points[npts_start+i,7]=1
        
        stacked_points.extend(points)
    
    stacked_points = np.array(stacked_points)
    stacked_points = downsample(stacked_points, DOWNSAMPLE_RESOLUTION)
    
    equalized_idx = []
    unequalized_idx = []
    equalized_map = {}
    point_voxels = [tuple(v) for v in np.round(stacked_points[:,:3]/VOXEL_RESOLUTION).astype(int)]
    for i in range(len(stacked_points)):
        k = point_voxels[i]
        if not k in equalized_map:
            equalized_map[k] = len(equalized_idx)
            equalized_idx.append(i)
        unequalized_idx.append(equalized_map[k])
    points = stacked_points[equalized_idx, :]
    point_voxels = [tuple(v) for v in np.round(points[:,:3]/VOXEL_RESOLUTION).astype(int)]
    obj_id = points[:, 3]
    cls_id = points[:, 4]
    new_obj_id = np.zeros(len(obj_id), dtype=int)

    # connected components to label unassigned obj IDs
    original_obj_id = set(points[:, 3]) - set([0])
    cluster_id = 1
    for i in original_obj_id:
        new_obj_id[obj_id == i] = cluster_id
        cluster_id += 1 

    edges = []
    for i in range(len(point_voxels)):
        if obj_id[i] > 0:
            continue
        k = point_voxels[i]
        for d in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
            if d!=(0,0,0):
                kk = (k[0]+d[0], k[1]+d[1], k[2]+d[2])
                if kk in equalized_map and cls_id[i] == cls_id[equalized_map[kk]]:
                    edges.append([i, equalized_map[kk]])
    G = nx.Graph(edges)
    clusters = nx.connected_components(G)
    clusters = [list(c) for c in clusters]
    for i in range(len(clusters)):
        if len(clusters[i]) > MIN_CLUSTER:
            new_obj_id[clusters[i]] = cluster_id
            cluster_id += 1

    stacked_points[:, 3] = new_obj_id[unequalized_idx]
    stacked_points = stacked_points[stacked_points[:, 3] > 0, :]

    print('Processing %d points from %s'%(len(stacked_points), scan_names[offset][:]))
    all_points.extend(stacked_points)
    counts.append(len(stacked_points))
    stacked_points = []

    offset += 1

# while offset < frames:
#     print(offset)

#     scan = np.fromfile(scan_names[offset], dtype=np.float32)
#     scan = scan.reshape((-1, 4))

#     # get XYZ in world coordinate frame
#     xyz_local = scan[:, 0:3]
#     R_world_local = poses[offset][:3, :3]
#     t_world_local = poses[offset][:3, 3]
#     xyz_world = xyz_local.dot(R_world_local.T) + t_world_local
    
#     # get point labels
#     label = np.fromfile(label_names[offset], dtype=np.uint32)
#     obj_id = [l >> 16 for l in label]
#     cls_id = [l & 0xFFFF for l in label]
        
#     trk_id = []
#     for i in obj_id:
#         trk_id.append(obj_ids_tr[obj_ids_uq.index(i)])
    
#     # stack in Nx8 array
#     points = np.zeros((len(xyz_world), 8))
#     points[:, :3] = xyz_local.dot(R_world_local.T) #xyz_world
#     points[:, 3] = obj_id
#     points[:, 4] = cls_id
#     points[:, 5] = offset
#     points[:, 6] = trk_id
#     points[:, 7] = 0
    
#     # Artificial sphere 1
#     npts_sphere1 = 300
#     sphere1_geom = sample_sphere(1,[1.5+(np.sin(float(offset)*0.1)*2.0), 5.0+(np.cos(float(offset)*0.1)*2.0), 1.0],npts_sphere1)
#     sphere1 = np.zeros((npts_sphere1, 8))
#     for i in range(npts_sphere1):
#         sphere1[i,0:3]=sphere1_geom[i][0:3]
#         sphere1[i,3]=5001
#         sphere1[i,4]=-1
#         sphere1[i,5]=offset
#         sphere1[i,6]=5001
#         sphere1[i,7]=1
        
#     # Artificial sphere 2
#     npts_sphere2 = 300
#     sphere2_geom = sample_sphere(1.33,[1.5+(np.sin(float(-offset)*0.1)*4.0), 5.0+(np.cos(float(-offset)*0.1)*4.0), 1.0],npts_sphere2)
#     sphere2 = np.zeros((npts_sphere2, 8))
#     for i in range(npts_sphere2):
#         sphere2[i,0:3]=sphere2_geom[i][0:3]
#         sphere2[i,3]=5002
#         sphere2[i,4]=-1
#         sphere2[i,5]=offset
#         sphere2[i,6]=5002
#         sphere2[i,7]=1

#     # Artificial cube 1
#     npts_cube1 = 300
#     cube1_geom = sample_cube(1.33,[1.5+(np.sin(float(offset)*0.1)*6.0), 5.0+(np.cos(float(offset)*0.1)*6.0), 1.0],npts_cube1)
#     cube1 = np.zeros((npts_cube1, 8))
#     for i in range(npts_cube1):
#         cube1[i,0:3]=cube1_geom[i][0:3]
#         cube1[i,3]=5003
#         cube1[i,4]=-1
#         cube1[i,5]=offset
#         cube1[i,6]=5003
#         cube1[i,7]=1
    
#     points = np.array(points)
# #     print(points.shape)
# #     print(sphere1.shape)
#     points = np.concatenate((points,sphere1))
#     points = np.concatenate((points,sphere2))
#     points = np.concatenate((points,cube1))
#     points = downsample(points, DOWNSAMPLE_RESOLUTION)

#     print('Processing %d points from %s'%(len(points), scan_names[offset][:]))
#     all_points.extend(points)
#     counts.append(len(points))

#     offset += 1
    
# fig1,ax1 = plt.subplots(figsize=(10, 10))

# rot1=45
# rot2=45
# rot3=0
# r = R.from_euler('zyx', [rot1, rot2, rot3], degrees=True)

# ims = []
# id_sta=0
# id_end=counts[0]
# for w in range(len(counts)):
#     if w > 0:
#         id_sta+=counts[w-1]
#         id_end=id_sta+counts[w]
#     points_fr=all_points[id_sta:id_end]
#     npt=len(points_fr)
#     rgb=np.zeros((1000,1000,3),dtype=np.uint8)
#     for k in range(npt):
#         vr=r.apply(points_fr[k][0:3])
#         obj_id_pt = int(points_fr[k][3])
#         trk_id_pt = int(points_fr[k][6])
    
#         x=int(vr[0]*15)+500
#         y=int(vr[1]*15)+500
#         if (x>=0) and (x<1000) and (y>=0) and (y<1000):
#             if trk_id_pt<5000:
#                 if not(tracks[trk_id_pt].moving):
#                     rgb[x,y,0]=0
#                     rgb[x,y,1]=int(colors[obj_id_pt,1])
#                     rgb[x,y,2]=int(colors[obj_id_pt,2])
#                 else:
#                     rgb[x,y,0]=255 #int(colors[obj_id_pt,0])
#                     rgb[x,y,1]=0
#                     rgb[x,y,2]=0
#             else:
#                 rgb[x,y,0]=255
#                 rgb[x,y,1]=0
#                 rgb[x,y,2]=0
            
#     im = ax1.imshow(rgb, animated=True)
#     if w == 0:
#         ax1.imshow(rgb)  # show an initial one first
#     ims.append([im])

# ani = animation.ArtistAnimation(fig1, ims, interval=50, blit=True, repeat_delay=1000)
# ani.save("KITTI_" + sequence + "_MOTION_MOD.gif")
    
# print('Unique Tracks: ',len(tracks))
# for i in range(len(tracks)):
#     print('Track # ',i,' with obj_id: ',tracks[i].obj_id)
#     print(tracks[i].frames)
#     print(tracks[i].centroids)


    
#     if offset % INTERVAL == INTERVAL - 1:
#         stacked_points = np.array(stacked_points)
#         stacked_points = downsample(stacked_points, DOWNSAMPLE_RESOLUTION)

#         # get equalized resolution for connected components
#         
#         print('Creating data sample with %d->%d points %d->%d objects' % (len(stacked_points), len(points), len(original_obj_id), len(set(new_obj_id))))
#         all_points.extend(stacked_points)
#         count.append(len(stacked_points))
#         offset += SKIP * INTERVAL + 1
#         stacked_points = []
#     else:
#          offset += 1

h5_fout = h5py.File('Z:\PHD\pcseg\KITTI_MOD\KITTI_I20_' + sequence + '.h5','w')
h5_fout.create_dataset('points', data=all_points, compression='gzip', compression_opts=4, dtype=np.float32)
h5_fout.create_dataset('count_room', data=counts, compression='gzip', compression_opts=4, dtype=np.int32)
h5_fout.close()