import motrackers
import tensorflow as tf
import h5py
import itertools
import time
import scipy
#from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score

classes_kitti = [''] * 260
classes_kitti[0] = "unlabeled"
classes_kitti[1] = "outlier"
classes_kitti[10] = "car"
classes_kitti[11] = "bicycle"
classes_kitti[13] = "bus"
classes_kitti[15] = "motorcycle"
classes_kitti[16] = "on-rails"
classes_kitti[18] = "truck"
classes_kitti[20] = "other-vehicle"
classes_kitti[30] = "person"
classes_kitti[31] = "bicyclist"
classes_kitti[32] = "motorcyclist"
classes_kitti[40] = "road"
classes_kitti[44] = "parking"
classes_kitti[48] = "sidewalk"
classes_kitti[49] = "other-ground"
classes_kitti[50] = "building"
classes_kitti[51] = "fence"
classes_kitti[52] = "other-structure"
classes_kitti[60] = "lane-marking"
classes_kitti[70] = "vegetation"
classes_kitti[71] = "trunk"
classes_kitti[72] = "terrain"
classes_kitti[80] = "pole"
classes_kitti[81] = "traffic-sign"
classes_kitti[99] = "other-object"
classes_kitti[252] = "moving-car"
classes_kitti[253] = "moving-bicyclist"
classes_kitti[254] = "moving-person"
classes_kitti[255] = "moving-motorcyclist"
classes_kitti[256] = "moving-on-rails"
classes_kitti[257] = "moving-bus"
classes_kitti[258] = "moving-truck"
classes_kitti[259] = "moving-other-vehicle"

agg_nmi = []
agg_ami = []
agg_ars = []
agg_prc = []
agg_rcl = []
agg_iou = []
comp_time_analysis = {
	'feature': [],
	'net': [],
	'neighbor': [],
	'inlier': [],
	'current_net' : [],
	'current_neighbor' : [],
	'current_inlier' : [],
	'iter_net' : [],
	'iter_neighbor' : [],
	'iter_inlier' : [],
}

def saveCSV(filename, points):
	f = open(filename,'w')
	for p in points:
		f.write("%f,%f,%f,%d,%d,%d\n"%(p[0],p[1],p[2],p[3],p[4],p[5]))
	f.close()
	print('Saved to %s: (%d points)'%(filename, len(points)))
    
def saveCSVM(filename, points):
	f = open(filename,'w')
	for p in points:
		f.write("%f,%f,%f,%d,%d,%d,%d,%f,%f,%f,%d\n"%(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10]))
	f.close()
	print('Saved to %s: (%d points)'%(filename, len(points)))


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

all_points,all_obj_id,all_cls_id,all_frm_id,all_motion = loadFromH5ModV2('Z:\PHD\pcseg\KITTI_MOD\KITTI_I20_06.h5')
# all_points,all_obj_id,all_cls_id,all_frm_id,all_motion = loadFromH5ModV2('Z:\PHD\pcseg\KITTI_MOD\KITTI_I20_01.h5')

NUM_INLIER_POINT = 512
NUM_NEIGHBOR_POINT = 512
TRAIN_AREA = [1]
VAL_AREA = [3]
FEATURE_SIZE = 11
LITE = 1
resolution = 0.3
add_threshold = 0.5
rmv_threshold = 0.5
cluster_threshold = 10
save_results = True
save_id = 0

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.compat.v1.Session(config=config)

net = LrgNet(1, 1, NUM_INLIER_POINT, NUM_NEIGHBOR_POINT, FEATURE_SIZE, LITE)

MODEL_PATH = 'Z:/PHD/pcseg/models4/lrgnet_model%s_xyz_mod.ckpt'%VAL_AREA[0]

saver = tf.compat.v1.train.Saver()
saver.restore(sess, MODEL_PATH)
print('Restored from %s'%MODEL_PATH)

file=open('Z:/PHD/pcseg/results/lrg/MOT_300K_4S_TR_1_LITE_06_0p3/MOT_300K_4S_TR_1_LITE_06_0p3.csv','w')

for room_id in range(len(all_points)):
	print('Room: ',room_id)
	unequalized_points = all_points[room_id]
	obj_id = all_obj_id[room_id]
	cls_id = all_cls_id[room_id]
	frm_id = all_frm_id[room_id]
	motion = all_motion[room_id]

	#equalize resolution
	t1 = time.time()
	equalized_idx = []
	unequalized_idx = []
	equalized_map = {}
	normal_grid = {}
	for i in range(len(unequalized_points)):
		k = tuple(numpy.round(unequalized_points[i,:3]/resolution).astype(int))
		if not k in equalized_map:
			equalized_map[k] = len(equalized_idx)
			equalized_idx.append(i)
		unequalized_idx.append(equalized_map[k])
		if not k in normal_grid:
			normal_grid[k] = []
		normal_grid[k].append(i)
	points = unequalized_points[equalized_idx]
	obj_id = obj_id[equalized_idx]
	cls_id = cls_id[equalized_idx]
	frm_id = frm_id[equalized_idx]
	motion = motion[equalized_idx]
	xyz = points[:,:3]
	room_coordinates = (xyz - xyz.min(axis=0)) / (xyz.max(axis=0) - xyz.min(axis=0))

	#compute normals
	normals = []
	curvatures = []
	for i in range(len(points)):
		k = tuple(numpy.round(points[i,:3]/resolution).astype(int))
		neighbors = []
		for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
			kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
			if kk in normal_grid:
				neighbors.extend(normal_grid[kk])
		accA = numpy.zeros((3,3))
		accB = numpy.zeros(3)
		for n in neighbors:
			p = unequalized_points[n,:3]
			accA += numpy.outer(p,p)
			accB += p
		cov = accA / len(neighbors) - numpy.outer(accB, accB) / len(neighbors)**2
		U,S,V = numpy.linalg.svd(cov)
		normals.append(numpy.fabs(V[2]))
		curvature = S[2] / (S[0] + S[1] + S[2])
		curvatures.append(numpy.fabs(curvature))
	curvatures = numpy.array(curvatures)
	curvatures = curvatures/curvatures.max()
	normals = numpy.array(normals)
	comp_time_analysis['feature'].append(time.time() - t1)
    
	points = numpy.hstack((xyz, room_coordinates, normals, curvatures.reshape(-1,1), frm_id.reshape(-1,1))).astype(numpy.float32)

	point_voxels = numpy.round(points[:,:3]/resolution).astype(int)
	cluster_label = numpy.zeros(len(points), dtype=int)
	motion_label = numpy.zeros(len(points), dtype=int)
	cluster_id = 1
	visited = numpy.zeros(len(point_voxels), dtype=bool)
	inlier_points = numpy.zeros((1, NUM_INLIER_POINT, FEATURE_SIZE), dtype=numpy.float32)
	neighbor_points = numpy.zeros((1, NUM_NEIGHBOR_POINT, FEATURE_SIZE), dtype=numpy.float32)
	input_add = numpy.zeros((1, NUM_NEIGHBOR_POINT), dtype=numpy.int32)
	input_remove = numpy.zeros((1, NUM_INLIER_POINT), dtype=numpy.int32)
	input_motion = numpy.zeros((1, NUM_INLIER_POINT+NUM_NEIGHBOR_POINT), dtype=numpy.int32)
	order = numpy.argsort(curvatures)
	#iterate over each object in the room
	#for seed_id in range(len(point_voxels)):
	for seed_id in numpy.arange(len(points))[order]:
		if visited[seed_id]:
			continue
		seed_voxel = point_voxels[seed_id]
		target_id = obj_id[seed_id]
		target_class = classes_kitti[cls_id[numpy.nonzero(obj_id==target_id)[0][0]]]
		gt_mask = obj_id==target_id
		obj_voxels = point_voxels[gt_mask]
		obj_voxel_set = set([tuple(p) for p in obj_voxels])
		original_minDims = obj_voxels.min(axis=0)
		original_maxDims = obj_voxels.max(axis=0)
		currentMask = numpy.zeros(len(points), dtype=bool)
		currentMask[seed_id] = True
		minDims = seed_voxel.copy()
		maxDims = seed_voxel.copy()
		seqMinDims = minDims
		seqMaxDims = maxDims
		steps = 0
		stuck = 0
		maskLogProb = 0

		#perform region growing
		while True:

			def stop_growing(reason):
				global cluster_id, start_time
				visited[currentMask] = True
				if numpy.sum(currentMask) > cluster_threshold:
					cluster_label[currentMask] = cluster_id
					cluster_id += 1
					iou = 1.0 * numpy.sum(numpy.logical_and(gt_mask,currentMask)) / numpy.sum(numpy.logical_or(gt_mask,currentMask))
					#print('room %d target %3d %.4s: step %3d %4d/%4d points IOU %.3f add %.3f rmv %.3f %s'%(room_id, target_id, target_class, steps, numpy.sum(currentMask), numpy.sum(gt_mask), iou, add_acc, rmv_acc, reason))
					strout='room %d target %3d %.4s: step %3d %4d/%4d points IOU %.3f add %.3f rmv %.3f mot %.3f %s'%(room_id, target_id, target_class, steps, numpy.sum(currentMask), numpy.sum(gt_mask), iou, add_acc, rmv_acc, mot_acc, reason)
					strfil='%d,%3d,%.4s,%3d,%4d,%4d,%.3f,%.3f,%.3f,%.3f,%s'%(room_id, target_id, target_class, steps, numpy.sum(currentMask), numpy.sum(gt_mask), iou, add_acc, rmv_acc, mot_acc, reason)
					print(strout)
					file.write(strfil)
					file.write('\n')
                    
			#determine the current points and the neighboring points
			t = time.time()
			currentPoints = points[currentMask, :].copy()
			newMinDims = minDims.copy()	
			newMaxDims = maxDims.copy()	
			newMinDims -= 1
			newMaxDims += 1
			mask = numpy.logical_and(numpy.all(point_voxels>=newMinDims,axis=1), numpy.all(point_voxels<=newMaxDims, axis=1))
			mask = numpy.logical_and(mask, numpy.logical_not(currentMask))
			mask = numpy.logical_and(mask, numpy.logical_not(visited))
			expandPoints = points[mask, :].copy()
			expandClass = obj_id[mask] == target_id
			expandClass_motion = motion[mask]
			rejectClass = obj_id[currentMask] != target_id
			rejectClass_motion = motion[currentMask] 

			if len(expandPoints)==0: #no neighbors (early termination)
				stop_growing('noneighbor')
				break

			if len(currentPoints) >= NUM_INLIER_POINT:
				subset = numpy.random.choice(len(currentPoints), NUM_INLIER_POINT, replace=False)
			else:
				subset = list(range(len(currentPoints))) + list(numpy.random.choice(len(currentPoints), NUM_INLIER_POINT-len(currentPoints), replace=True))
			center = numpy.median(currentPoints, axis=0)
			expandPoints = numpy.array(expandPoints)
			expandPoints[:,:2] -= center[:2]
			expandPoints[:,6:] -= center[6:]
			inlier_points[0,:,:] = currentPoints[subset, :]
			inlier_points[0,:,:2] -= center[:2]
			inlier_points[0,:,6:] -= center[6:]
			input_remove[0,:] = numpy.array(rejectClass)[subset]
            
			input_motion[0,:NUM_INLIER_POINT]=numpy.array(rejectClass_motion)[subset]            
            
			if len(expandPoints) >= NUM_NEIGHBOR_POINT:
				subset = numpy.random.choice(len(expandPoints), NUM_NEIGHBOR_POINT, replace=False)
			else:
				subset = list(range(len(expandPoints))) + list(numpy.random.choice(len(expandPoints), NUM_NEIGHBOR_POINT-len(expandPoints), replace=True))
			neighbor_points[0,:,:] = numpy.array(expandPoints)[subset, :]
			input_add[0,:] = numpy.array(expandClass)[subset]
            
			input_motion[0,NUM_INLIER_POINT:]=numpy.array(expandClass_motion)[subset]            
            
			comp_time_analysis['current_neighbor'].append(time.time() - t)
			t = time.time()
			ls, add,add_acc, rmv,rmv_acc, mot,mot_acc = sess.run([net.loss, net.add_output, net.add_acc, net.remove_output, net.remove_acc, net.motion_output, net.motion_acc],
				{net.inlier_pl:inlier_points, net.neighbor_pl:neighbor_points, net.add_mask_pl:input_add, net.remove_mask_pl:input_remove, net.motion_mask_pl:input_motion})
			comp_time_analysis['current_net'].append(time.time() - t)
			t = time.time()

			add_conf = scipy.special.softmax(add[0], axis=-1)[:,1]
			rmv_conf = scipy.special.softmax(rmv[0], axis=-1)[:,1]
			mot_conf = scipy.special.softmax(mot[0], axis=-1)[:,1]
#			add_mask = add_conf > add_threshold
#			rmv_mask = rmv_conf > rmv_threshold
			add_mask = numpy.random.random(len(add_conf)) < add_conf
			rmv_mask = numpy.random.random(len(rmv_conf)) < rmv_conf
			mot_mask = numpy.random.random(len(mot_conf)) < mot_conf
#			add_mask = input_add[0].astype(bool)
#			rmv_mask = input_remove[0].astype(bool)
			addPoints = neighbor_points[0,:,:][add_mask]
			addPoints[:,:2] += center[:2]
			addVoxels = numpy.round(addPoints[:,:3]/resolution).astype(int)
			addSet = set([tuple(p) for p in addVoxels])
			rmvPoints = inlier_points[0,:,:][rmv_mask]
			rmvPoints[:,:2] += center[:2]
			rmvVoxels = numpy.round(rmvPoints[:,:3]/resolution).astype(int)
			rmvSet = set([tuple(p) for p in rmvVoxels])

			motPoints = numpy.vstack((inlier_points[0,:,:],neighbor_points[0,:,:]))[mot_mask]
			motPoints[:,:2] += center[:2]
			motVoxels = numpy.round(motPoints[:,:3]/resolution).astype(int)
			motSet = set([tuple(p) for p in motVoxels])

			updated = False
			iou = 1.0 * numpy.sum(numpy.logical_and(gt_mask,currentMask)) / numpy.sum(numpy.logical_or(gt_mask,currentMask))
#			print('%d/%d points %d outliers %d/%d add %d/%d rmv %.2f iou'%(numpy.sum(numpy.logical_and(currentMask, gt_mask)), numpy.sum(gt_mask),
#				numpy.sum(numpy.logical_and(gt_mask==0, currentMask)), len(addSet), len(expandPoints), len(rmvSet), len(currentPoints), iou))
			for i in range(len(point_voxels)):
				if not currentMask[i] and tuple(point_voxels[i]) in addSet:
					currentMask[i] = True
					updated = True
				if tuple(point_voxels[i]) in rmvSet:
					currentMask[i] = False
				if tuple(point_voxels[i]) in motSet:                    
					motion_label[i] = 1
			steps += 1
			comp_time_analysis['current_inlier'].append(time.time() - t)

			if updated: #continue growing
				minDims = point_voxels[currentMask, :].min(axis=0)
				maxDims = point_voxels[currentMask, :].max(axis=0)
				if not numpy.any(minDims<seqMinDims) and not numpy.any(maxDims>seqMaxDims):
					if stuck >= 1:
						stop_growing('stuck')
						break
					else:
						stuck += 1
				else:
					stuck = 0
				seqMinDims = numpy.minimum(seqMinDims, minDims)
				seqMaxDims = numpy.maximum(seqMaxDims, maxDims)
			else: #no matching neighbors (early termination)
				stop_growing('noexpand')
				break

	#fill in points with no labels
	nonzero_idx = numpy.nonzero(cluster_label)[0]
	nonzero_points = points[nonzero_idx, :]
	filled_cluster_label = cluster_label.copy()
	for i in numpy.nonzero(cluster_label==0)[0]:
		d = numpy.sum((nonzero_points - points[i])**2, axis=1)
		closest_idx = numpy.argmin(d)
		filled_cluster_label[i] = cluster_label[nonzero_idx[closest_idx]]
	cluster_label = filled_cluster_label
	print('%d %d points: %.2fs' % (room_id, len(unequalized_points), time.time() - t1))

	#calculate statistics 
	gt_match = 0
	match_id = 0
	dt_match = numpy.zeros(cluster_label.max(), dtype=bool)
	cluster_label2 = numpy.zeros(len(cluster_label), dtype=int)
	room_iou = []
	unique_id, count = numpy.unique(obj_id, return_counts=True)
	for k in range(len(unique_id)):
		i = unique_id[numpy.argsort(count)][::-1][k]
		best_iou = 0
		for j in range(1, cluster_label.max()+1):
			if not dt_match[j-1]:
				iou = 1.0 * numpy.sum(numpy.logical_and(obj_id==i, cluster_label==j)) / numpy.sum(numpy.logical_or(obj_id==i, cluster_label==j))
				best_iou = max(best_iou, iou)
				if iou > 0.5:
					dt_match[j-1] = True
					gt_match += 1
					cluster_label2[cluster_label==j] = k+1
					break
		room_iou.append(best_iou)
	for j in range(1,cluster_label.max()+1):
		if not dt_match[j-1]:
			cluster_label2[cluster_label==j] = j + obj_id.max()
	prc = numpy.mean(dt_match)
	rcl = 1.0 * gt_match / len(set(obj_id))
	room_iou = numpy.mean(room_iou)

	nmi = 0#normalized_mutual_info_score(obj_id,cluster_label)
	ami = 0#adjusted_mutual_info_score(obj_id,cluster_label)
	ars = 0#adjusted_rand_score(obj_id,cluster_label)
	agg_nmi.append(nmi)
	agg_ami.append(ami)
	agg_ars.append(ars)
	agg_prc.append(prc)
	agg_rcl.append(rcl)
	agg_iou.append(room_iou)
	print("room %d NMI: %.2f AMI: %.2f ARS: %.2f PRC: %.2f RCL: %.2f IOU: %.2f"%(room_id, nmi,ami,ars, prc, rcl, room_iou))

	comp_time_analysis['neighbor'].append(sum(comp_time_analysis['current_neighbor']))
	comp_time_analysis['iter_neighbor'].extend(comp_time_analysis['current_neighbor'])
	comp_time_analysis['current_neighbor'] = []
	comp_time_analysis['net'].append(sum(comp_time_analysis['current_net']))
	comp_time_analysis['iter_net'].extend(comp_time_analysis['current_net'])
	comp_time_analysis['current_net'] = []
	comp_time_analysis['inlier'].append(sum(comp_time_analysis['current_inlier']))
	comp_time_analysis['iter_inlier'].extend(comp_time_analysis['current_inlier'])
	comp_time_analysis['current_inlier'] = []

	print(cluster_label2.shape)
	print(len(unequalized_idx))
	#save point cloud results to file
	if save_results:
		#color_sample_state = numpy.random.RandomState(0)
		#obj_color = color_sample_state.randint(0,255,(numpy.max(cluster_label2)+1,3))
		#obj_color[0] = [100,100,100]
		#unequalized_points[:,3:6] = obj_color[cluster_label2,:][unequalized_idx]
		colors=numpy.random.randint(0,255,(numpy.max(cluster_label)+1,3))
		colors[0] = [100,100,100]
		for i in range(len(points)):
			points[i,3:6] = colors[cluster_label[i],:]
			points[i,6] = motion_label[i]
		#saveCSV('Z:/PHD/pcseg/results/lrg/MOT/%d.csv'%save_id, points)
		saveCSVM('Z:/PHD/pcseg/results/lrg/MOT_300K_4S_TR_1_LITE_06_0p3/M_%d.csv'%save_id, points)
		save_id += 1

print('NMI: %.2f+-%.2f AMI: %.2f+-%.2f ARS: %.2f+-%.2f PRC %.2f+-%.2f RCL %.2f+-%.2f IOU %.2f+-%.2f'%
(numpy.mean(agg_nmi), numpy.std(agg_nmi),numpy.mean(agg_ami),numpy.std(agg_ami),numpy.mean(agg_ars),numpy.std(agg_ars),
numpy.mean(agg_prc), numpy.std(agg_prc), numpy.mean(agg_rcl), numpy.std(agg_rcl), numpy.mean(agg_iou), numpy.std(agg_iou)))

file.close()