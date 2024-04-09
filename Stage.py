# STAGE DATA MODIFIED V2

import h5py
import itertools
import sys
import numpy as np

def loadFromH5ModV2(filename):
	f = h5py.File(filename,'r')
	all_points = f['points'][:]
	count_room = f['count_room'][:]
	tmp_points = []
	idp = 0
	for i in range(len(count_room)):
		tmp_points.append(all_points[idp:idp+count_room[i], :])
		idp += count_room[i]
	f.close()
	room = []
	labels = []
	class_labels = []
	frame_labels = []
	motion_labels = []
	for i in range(len(tmp_points)):
		room.append(tmp_points[i][:,:-5])
		labels.append(tmp_points[i][:,-5].astype(int))
		class_labels.append(tmp_points[i][:,-4].astype(int))
		frame_labels.append(tmp_points[i][:,-3].astype(int))
		motion_labels.append(tmp_points[i][:,-1].astype(int))
	return room, labels, class_labels, frame_labels, motion_labels

resolution = 0.3 #0.6 #0.1
SEED = 4
cluster_threshold = 10
max_points = 1024
save_id = 0
np.random.seed(0)
AREAS = ['KITTI_I20']

for AREA in AREAS:
	all_points,all_obj_id,all_cls_id,all_frm_id,all_motion = loadFromH5ModV2('Z:\PHD\pcseg\KITTI_MOD\%s_%02d.h5' % (AREA, SEED))
	stacked_points = []
	stacked_neighbor_points = []
	stacked_count = []
	stacked_neighbor_count = []
	stacked_remove = []
	stacked_add = []
	stacked_motion = []
	stacked_steps = []
	stacked_complete = []

	for room_id in range(30): #len(all_points)):
		print('Room ID: ',room_id)

		unequalized_points = all_points[room_id]
		obj_id = all_obj_id[room_id]
		cls_id = all_cls_id[room_id]
		frm_id = all_frm_id[room_id]
		motion = all_motion[room_id]
		#normalize frame IDs

		#equalize resolution
		equalized_idx = []
		equalized_set = set()
		normal_grid = {}
		for i in range(len(unequalized_points)):
			k = tuple(np.round(unequalized_points[i,:3]/resolution).astype(int))
			if not k in equalized_set:
				equalized_set.add(k)
				equalized_idx.append(i)
			if not k in normal_grid:
				normal_grid[k] = []
			normal_grid[k].append(i)
		# points -> XYZ + RGB
		points = unequalized_points[equalized_idx]
		obj_id = obj_id[equalized_idx]
		cls_id = cls_id[equalized_idx]
		frm_id = frm_id[equalized_idx]
		motion = motion[equalized_idx]
		xyz = points[:,:3]
		rgb = points[:,3:6]
		room_coordinates = (xyz - xyz.min(axis=0)) / (xyz.max(axis=0) - xyz.min(axis=0))

		#compute normals and curvatures
		normals = []
		curvatures = []
		for i in range(len(points)):
			k = tuple(np.round(points[i,:3]/resolution).astype(int))
			neighbors = []
			for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
				kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
				if kk in normal_grid:
					neighbors.extend(normal_grid[kk])
			accA = np.zeros((3,3))
			accB = np.zeros(3)
			for n in neighbors:
				p = unequalized_points[n,:3]
				accA += np.outer(p,p)
				accB += p
			cov = accA / len(neighbors) - np.outer(accB, accB) / len(neighbors)**2
			U,S,V = np.linalg.svd(cov)
			normals.append(np.fabs(V[2]))
			curvature = S[2] / (S[0] + S[1] + S[2])
			curvatures.append(np.fabs(curvature)) 
		normals = np.array(normals)
		curvatures = np.array(curvatures)
		curvatures = curvatures/curvatures.max()

		points = np.hstack((xyz, room_coordinates, normals, curvatures.reshape(-1,1), frm_id.reshape(-1,1))).astype(np.float32)

		point_voxels = np.round(points[:,:3]/resolution).astype(int)
		visited = np.zeros(len(point_voxels), dtype=bool)

		#iterate over each voxel in the room
		for seed_id in np.random.choice(range(len(points)), len(points), replace=False):
#		for seed_id in np.arange(len(points))[np.argsort(curvatures)]:
			if visited[seed_id]:
				continue
			target_id = obj_id[seed_id]
			obj_voxels = point_voxels[obj_id==target_id, :]
# 			print(obj_voxels)
			gt_mask = obj_id==target_id
			original_minDims = obj_voxels.min(axis=0)
			original_maxDims = obj_voxels.max(axis=0)
			#print('original',np.sum(gt_mask), original_minDims, original_maxDims)
			mask = np.logical_and(np.all(point_voxels>=original_minDims,axis=1), np.all(point_voxels<=original_maxDims, axis=1))
			originalScore = 1.0 * np.sum(np.logical_and(gt_mask,mask)) / np.sum(np.logical_or(gt_mask,mask))

			#initialize the seed voxel
			seed_voxel = point_voxels[seed_id]
			currentMask = np.zeros(len(points), dtype=bool)
			currentMask[seed_id] = True
			minDims = seed_voxel.copy()
			maxDims = seed_voxel.copy()
			steps = 0
			stuck = False
			add_mistake_prob = np.random.randint(2,5)*0.1
			remove_mistake_prob = np.random.randint(2,5)*0.1
			add_mistake_limit = np.inf
			remove_mistake_limit = np.inf
#			add_mistake_prob = 0.03
#			remove_mistake_prob = 0.01
#			add_mistake_limit = 30.0
#			remove_mistake_limit = 30.0

			#perform region growing for all frames
			frm_done=False

			while not(frm_done):

				#determine the current points and the neighboring points
				currentPoints = points[currentMask, :].copy()
				newMinDims = minDims.copy()	
				newMaxDims = maxDims.copy()	
				newMinDims -= 1
				newMaxDims += 1
				mask = np.logical_and(np.all(point_voxels>=newMinDims,axis=1), np.all(point_voxels<=newMaxDims, axis=1))
				mask = np.logical_and(mask, np.logical_not(currentMask))
				mask = np.logical_and(mask, np.logical_not(visited))

				#determine which points to accept
				expandPoints = points[mask, :].copy()
				expandClass = obj_id[mask] == target_id
				expandClass_motion = motion[mask]           
				mask_idx = np.nonzero(mask)[0]
				if stuck:
					expandID = mask_idx[expandClass]
				else:
#					mistake_sample = np.random.random(len(mask_idx)) < add_mistake_prob
					mistake_sample = np.random.random(len(mask_idx)) < min(add_mistake_prob, add_mistake_limit/(len(mask_idx)+1))
					expand_with_mistake = np.logical_xor(expandClass, mistake_sample)
					expandID = mask_idx[expand_with_mistake]

				#determine which points to reject
				rejectClass = obj_id[currentMask] != target_id
				rejectClass_motion = motion[currentMask] 
				mask_idx = np.nonzero(currentMask)[0]
				if stuck:
					rejectID = mask_idx[rejectClass]
				else:
#					mistake_sample = np.random.random(len(mask_idx)) < remove_mistake_prob
					mistake_sample = np.random.random(len(mask_idx)) < min(remove_mistake_prob, remove_mistake_limit/(len(mask_idx)+1))
					reject_with_mistake = np.logical_xor(rejectClass, mistake_sample)
					rejectID = mask_idx[reject_with_mistake]
          
				if (len(expandPoints) > 0) and (len(currentPoints) > 0):
					if len(currentPoints) <= max_points:
						stacked_points.append(currentPoints)
						stacked_count.append(len(currentPoints))
						stacked_remove.extend(rejectClass)
						stacked_motion.extend(rejectClass_motion)
					else:
						subset = np.random.choice(len(currentPoints), max_points, replace=False)
						stacked_points.append(currentPoints[subset])
						stacked_count.append(max_points)
						rejectClass = rejectClass[subset]
						stacked_remove.extend(rejectClass)
						rejectClass_motion = rejectClass_motion[subset]
						stacked_motion.extend(rejectClass_motion)
					if len(expandPoints) <= max_points:
						stacked_neighbor_points.append(np.array(expandPoints))
						stacked_neighbor_count.append(len(expandPoints))
						stacked_add.extend(expandClass)
						stacked_motion.extend(expandClass_motion)
					else:
						subset = np.random.choice(len(expandPoints), max_points, replace=False)
						stacked_neighbor_points.append(expandPoints[subset])
						stacked_neighbor_count.append(max_points)
						expandClass = expandClass[subset]
						stacked_add.extend(expandClass)
						expandClass_motion = expandClass_motion[subset]
						stacked_motion.extend(expandClass_motion)
					iou = 1.0 * np.sum(np.logical_and(currentMask, gt_mask)) / np.sum(np.logical_or(currentMask, gt_mask))
					stacked_complete.append(iou)
#					stacked_complete.append(np.logical_not(np.logical_or(np.any(rejectClass), np.any(expandClass))))
					steps += 1
					add_mistake_prob = max(add_mistake_prob-0.01, 0.00)
					remove_mistake_prob = max(remove_mistake_prob-0.01, 0.00)

				if np.all(currentMask == gt_mask): #completed
					visited[currentMask] = True
					stacked_steps.append(steps)
					print('AREA %s room %d target %d : %d steps %d/%d (%.2f/%.2f IOU)'%(str(AREA), room_id, target_id, steps, np.sum(currentMask), np.sum(gt_mask), iou, originalScore))
					frm_done = True
				else:
					if steps < 500 and (np.any(expandClass) or np.any(rejectClass)): #continue growing
						#has matching neighbors: expand in those directions
						#update current mask
						currentMask[expandID] = True
						if len(rejectID) < len(mask_idx):
							currentMask[rejectID] = False
						if np.sum(currentMask)>0:
							nextMinDims = point_voxels[currentMask, :].min(axis=0)
							nextMaxDims = point_voxels[currentMask, :].max(axis=0)
						if not np.any(nextMinDims<minDims) and not np.any(nextMaxDims>maxDims):
							stuck = True
							iou = 1.0 * np.sum(np.logical_and(currentMask, gt_mask)) / np.sum(np.logical_or(currentMask, gt_mask))
							minDims = nextMinDims
							maxDims = nextMaxDims
					else: #no matching neighbors (early termination)
						if np.sum(currentMask) > cluster_threshold:
							visited[currentMask] = True
							stacked_steps.append(steps)
							print('AREA %s room %d target %d : %d steps %d/%d (%.2f/%.2f IOU)'%(str(AREA), room_id, target_id, steps, np.sum(currentMask), np.sum(gt_mask), iou, originalScore))
						frm_done = True

	for i in range(len(stacked_points)):
		center = np.median(stacked_points[i][:,:2], axis=0)
		feature_center = np.median(stacked_points[i][:,6:], axis=0)
		stacked_points[i][:,:2] -= center
		stacked_points[i][:,6:] -= feature_center
		if len(stacked_neighbor_points[i]) > 0:
			stacked_neighbor_points[i][:,:2] -= center
			stacked_neighbor_points[i][:,6:] -= feature_center

	#h5_fout = h5py.File('Z:\PHD\pcseg\multiseed\seed%d_area%s.h5'%(SEED,AREA),'w')
	h5_fout = h5py.File('Z:\PHD\pcseg\multiseed_res0p3\seed%d_area%s.h5'%(SEED,AREA),'w')
	h5_fout.create_dataset( 'points', data=np.vstack(stacked_points), compression='gzip', compression_opts=4, dtype=np.float32)
	h5_fout.create_dataset( 'count', data=stacked_count, compression='gzip', compression_opts=4, dtype=np.int32)
	h5_fout.create_dataset( 'neighbor_points', data=np.vstack(stacked_neighbor_points), compression='gzip', compression_opts=4, dtype=np.float32)
	h5_fout.create_dataset( 'neighbor_count', data=stacked_neighbor_count, compression='gzip', compression_opts=4, dtype=np.int32)
	h5_fout.create_dataset( 'add', data=stacked_add, compression='gzip', compression_opts=4, dtype=np.int32)
	h5_fout.create_dataset( 'remove', data=stacked_remove, compression='gzip', compression_opts=4, dtype=np.int32)
	h5_fout.create_dataset( 'motion', data=stacked_motion, compression='gzip', compression_opts=4, dtype=np.int32)
	h5_fout.create_dataset( 'steps', data=stacked_steps, compression='gzip', compression_opts=4, dtype=np.int32)
	h5_fout.create_dataset( 'complete', data=stacked_complete, compression='gzip', compression_opts=4, dtype=np.float32)
	h5_fout.close()