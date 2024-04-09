# Track Creation

import math
import numpy as np
from scipy.optimize import linear_sum_assignment
import os
import pandas as pd

series = 'MOT_300K_4S_TR_1_LITE_06_0p3'

csv_path = 'Z:/PHD/pcseg/results/lrg/'+series
csv_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(csv_path)) for f in fn]
csv_names.sort()

N_frames = len(csv_names)

f = open('Z:/PHD/pcseg/results/'+series+'_SEGSCENT.csv','w')


for i in range(N_frames):
    
    cloud = pd.read_csv(csv_names[i],header=None).to_numpy()
    npt = cloud.shape[0]
    
    
    segs=set()
    for k in range(npt):
        rr=cloud[k,3]
        gg=cloud[k,4]
        bb=cloud[k,5]
        rgbt=(rr,gg,bb)
        segs.add(rgbt)
        
    segs=list(segs)
    centroids=np.zeros((len(segs),3),dtype=np.float64)
    segcounts=np.zeros((len(segs)),dtype=np.int32)

    for k in range(npt):
        cx=cloud[k,0]
        cy=cloud[k,1]
        cz=cloud[k,2]
        rr=cloud[k,3]
        gg=cloud[k,4]
        bb=cloud[k,5]
        rgbt=(rr,gg,bb)
        indx=segs.index(rgbt)
        centroids[indx,0]+=cx
        centroids[indx,1]+=cy
        centroids[indx,2]+=cz
        segcounts[indx]+=1

    for k in range(len(segs)):
        centroids[k,0]/=float(segcounts[k])
        centroids[k,1]/=float(segcounts[k])
        centroids[k,2]/=float(segcounts[k])
        f.write("%d,%d,%d,%d,%d,%f,%f,%f\n"%(i,segs[k][0],segs[k][1],segs[k][2],segcounts[k],centroids[k,0],centroids[k,1],centroids[k,2]))

    print(i,npt,len(segs))
    
f.close()

csv_fn = 'Z:/PHD/pcseg/results/'+series+'_SEGSCENT.csv'

segs = pd.read_csv(csv_fn,header=None).to_numpy()
nframes = int(np.max(segs[:,0]))
N = segs.shape[0]
newcols = np.zeros((N,3),dtype=np.int32)

for i in range(N):
    newcols[i,0]=int(segs[i,1])
    newcols[i,1]=int(segs[i,2])
    newcols[i,2]=int(segs[i,3])

for i in range(nframes-1):
    indx1=i
    indx2=i+1
    s1_idx=segs[:,0]==indx1
    s2_idx=segs[:,0]==indx2
    ss1=segs[s1_idx]
    ss2=segs[s2_idx]
    s1_st=s1_idx.tolist().index(True)
    s2_st=s2_idx.tolist().index(True)
    N1=ss1.shape[0]
    N2=ss2.shape[0]
    print(indx1,indx2,N1,N2,s1_st,s2_st)
    CM=np.zeros((N1,N2),dtype=np.float64)
    for k in range(N1):
        for m in range(N2):
            CM[k,m]=np.linalg.norm([ss1[k,5]-ss2[m,5],ss1[k,6]-ss2[m,6],ss1[k,7]-ss2[m,7]])
    row_ind, col_ind = linear_sum_assignment(CM)
    for k in range(len(col_ind)):
        col_i = col_ind[k]
        row_i = row_ind[k]
        newcols[s2_st+col_i,0]=newcols[s1_st+row_i,0]
        newcols[s2_st+col_i,1]=newcols[s1_st+row_i,1]
        newcols[s2_st+col_i,2]=newcols[s1_st+row_i,2]
    print(row_ind,col_ind)
    
f = open('Z:/PHD/pcseg/results/'+series+'_SEGSCENT_NC.csv','w')
for i in range(N):
    f.write("%d,%d,%d\n"%(newcols[i,0],newcols[i,1],newcols[i,2]))
f.close()