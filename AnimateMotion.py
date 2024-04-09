import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os
import matplotlib.animation as animation
import pandas as pd
from PIL import Image

r2d=180/np.pi
d2r=np.pi/180

fig2,ax2 = plt.subplots(figsize=(10, 10))
rot1=45
rot2=45
rot3=0
r = R.from_euler('zyx', [rot1, rot2, rot3], degrees=True)


csv_path = 'Z:/PHD/pcseg/results/lrg/MOT_300K_4S_TR_1_LITE_06_0p3/'
csv_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(csv_path)) for f in fn]
csv_names.sort()

# get poses
sequence ='01'
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




scale = 10
ims = []
imgsz = 1000

N_frames = len(csv_names)

for i in range(N_frames):
    
    R_world_local = poses[i][:3, :3]
    t_world_local = poses[i][:3, 3]
    R_local_world = np.linalg.inv(R_world_local)
    
    print(i)
    rgb=np.zeros((imgsz,imgsz,3),dtype=np.uint8)
    
    cloud = pd.read_csv(csv_names[i],header=None).to_numpy()
    npt = cloud.shape[0]
    print(npt)
    
    for k in range(npt):
        cx=cloud[k,0]
        cy=cloud[k,1]
        cz=cloud[k,2]
        xyz_w=np.asarray([cx,cy,cz])
        xyz_l=xyz_w - t_world_local
        xyz_l=xyz_l.dot(R_local_world.T) 
        
        vr=r.apply([xyz_l[0],xyz_l[1],xyz_l[2]])
        x=int(vr[0]*scale)+int(imgsz/2)
        y=int(vr[1]*scale)+int(imgsz/2)
        if (x>=1) and (x<imgsz-1) and (y>=1) and (y<imgsz-1):
            motion=cloud[k,6]
            if motion>0:
                rr=255
                gg=0
                bb=0
            else:
                rr=100
                gg=100
                bb=100
            rgb[x,y,0]=rr
            rgb[x,y,1]=gg
            rgb[x,y,2]=bb
            rgb[x+1,y,0]=rr
            rgb[x+1,y,1]=gg
            rgb[x+1,y,2]=bb
            rgb[x-1,y,0]=rr
            rgb[x-1,y,1]=gg
            rgb[x-1,y,2]=bb
            rgb[x,y+1,0]=rr
            rgb[x,y+1,1]=gg
            rgb[x,y+1,2]=bb
            rgb[x,y-1,0]=rr
            rgb[x,y-1,1]=gg
            rgb[x,y-1,2]=bb

    im = ax2.imshow(rgb, animated=True)
    ims.append([im])
    
ani = animation.ArtistAnimation(fig2, ims, interval=200, blit=True, repeat_delay=1000)
ani.save("Z:/PHD/pcseg/results/MOT_300K_4S_TR_1_LITE_06_0p6_2.gif")