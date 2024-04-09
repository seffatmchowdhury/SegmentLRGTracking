import sys
import time

MAX_DATA = 300000

BATCH_SIZE = 150
NUM_INLIER_POINT = 512
NUM_NEIGHBOR_POINT = 512
MAX_EPOCH = 50
VAL_STEP = 5
TRAIN_AREA = [1,4,6,7]
VAL_AREA = [3]
FEATURE_SIZE = 11 #13
MULTISEED = 2
seeds=[3]
LITE = 1
initialized = False
cross_domain = False
numpy.random.seed(0)
numpy.set_printoptions(2,linewidth=100,suppress=True,sign=' ')

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.compat.v1.Session(config=config)
net = LrgNet(BATCH_SIZE, 1, NUM_INLIER_POINT, NUM_NEIGHBOR_POINT, FEATURE_SIZE, LITE)
saver = tf.compat.v1.train.Saver()

MODEL_PATH = 'Z:/PHD/pcseg/models4/lrgnet_model%s_xyz_mod.ckpt'%VAL_AREA[0]

epoch_time = []

init = tf.compat.v1.global_variables_initializer()
sess.run(init, {})
for epoch in range(MAX_EPOCH):

	if not initialized or MULTISEED > 1:
		initialized = True
		train_inlier_points, train_inlier_count, train_neighbor_points, train_neighbor_count, train_add, train_remove, train_motion = [], [], [], [], [], [], []
		val_inlier_points, val_inlier_count, val_neighbor_points, val_neighbor_count, val_add, val_remove, val_motion = [], [], [], [], [], [], []


			f = h5py.File('Z:/PHD/pcseg/multiseed/seed%d_areaKITTI_I20.h5'%(AREA), 'r')
			print('Loading %s ...'%f.filename)
			if VAL_AREA is not None and AREA in VAL_AREA:
				count = f['count'][:MAX_DATA]
				val_inlier_count.extend(count)
				points = f['points'][:]
				remove = f['remove'][:]
				idp = 0
				for i in range(len(count)):
					val_inlier_points.append(points[idp:idp+count[i], :FEATURE_SIZE])
					val_remove.append(remove[idp:idp+count[i]])
					idp += count[i]
				neighbor_count = f['neighbor_count'][:MAX_DATA]
				val_neighbor_count.extend(neighbor_count)
				neighbor_points = f['neighbor_points'][:]
				add = f['add'][:]
				idp = 0
				for i in range(len(neighbor_count)):
					val_neighbor_points.append(neighbor_points[idp:idp+neighbor_count[i], :FEATURE_SIZE])
					val_add.append(add[idp:idp+neighbor_count[i]])
					idp += neighbor_count[i]
				motion = f['motion'][:]
				idp = 0
				for i in range(len(count)):
					val_motion.append(motion[idp:(idp+neighbor_count[i]+count[i])])
					idp += count[i] + neighbor_count[i]
			if AREA in TRAIN_AREA:
				count = f['count'][:MAX_DATA]
				train_inlier_count.extend(count)
				points = f['points'][:]
				remove = f['remove'][:]
				idp = 0
				for i in range(len(count)):
					train_inlier_points.append(points[idp:idp+count[i], :FEATURE_SIZE])
					train_remove.append(remove[idp:idp+count[i]])
					idp += count[i]
				neighbor_count = f['neighbor_count'][:MAX_DATA]
				train_neighbor_count.extend(neighbor_count)
				neighbor_points = f['neighbor_points'][:]
				add = f['add'][:]
				idp = 0
				for i in range(len(neighbor_count)):
					train_neighbor_points.append(neighbor_points[idp:idp+neighbor_count[i], :FEATURE_SIZE])
					train_add.append(add[idp:idp+neighbor_count[i]])
					idp += neighbor_count[i]
				motion = f['motion'][:]
				idp = 0
				for i in range(len(count)):
					train_motion.append(motion[idp:(idp+neighbor_count[i]+count[i])])
					idp += count[i] + neighbor_count[i]
			if FEATURE_SIZE is None: 
				FEATURE_SIZE = points.shape[1]
			f.close()

		#filter out instances where the neighbor array is empty
		train_inlier_points = [train_inlier_points[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		train_inlier_count = [train_inlier_count[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		train_neighbor_points = [train_neighbor_points[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		train_add = [train_add[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		train_remove = [train_remove[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		train_neighbor_count = [train_neighbor_count[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		train_motion = [train_motion[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		val_inlier_points = [val_inlier_points[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		val_inlier_count = [val_inlier_count[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		val_neighbor_points = [val_neighbor_points[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		val_add = [val_add[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		val_remove = [val_remove[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		val_neighbor_count = [val_neighbor_count[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		val_motion = [val_motion[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		if len(train_inlier_points)==0:
			continue
		print('train',len(train_inlier_points),train_inlier_points[0].shape, len(train_neighbor_points))
		print('val',len(val_inlier_points), len(val_neighbor_points))

	idx = numpy.arange(len(train_inlier_points))
	numpy.random.shuffle(idx)
	inlier_points = numpy.zeros((BATCH_SIZE, NUM_INLIER_POINT, FEATURE_SIZE))
	neighbor_points = numpy.zeros((BATCH_SIZE, NUM_NEIGHBOR_POINT, FEATURE_SIZE))
	input_add = numpy.zeros((BATCH_SIZE, NUM_NEIGHBOR_POINT), dtype=numpy.int32)
	input_remove = numpy.zeros((BATCH_SIZE, NUM_INLIER_POINT), dtype=numpy.int32)
	input_motion = numpy.zeros((BATCH_SIZE, NUM_INLIER_POINT+NUM_NEIGHBOR_POINT), dtype=numpy.int32)

	loss_arr = []
	add_prc_arr = []
	add_rcl_arr = []
	rmv_prc_arr = []
	rmv_rcl_arr = []
	mot_prc_arr = []
	mot_rcl_arr = []
	num_batches = int(len(train_inlier_points) / BATCH_SIZE)
	start_time = time.time()
	for batch_id in range(num_batches):
		start_idx = batch_id * BATCH_SIZE
		end_idx = (batch_id + 1) * BATCH_SIZE
		for i in range(BATCH_SIZE):
			points_idx = idx[start_idx+i]
			N = train_inlier_count[points_idx]
			if N >= NUM_INLIER_POINT:
				subset = numpy.random.choice(N, NUM_INLIER_POINT, replace=False)
			else:
				subset = list(range(N)) + list(numpy.random.choice(N, NUM_INLIER_POINT-N, replace=True))
			inlier_points[i,:,:] = train_inlier_points[points_idx][subset, :]
			input_remove[i,:] = train_remove[points_idx][subset]
			N = train_neighbor_count[points_idx]
			if N >= NUM_NEIGHBOR_POINT:
				subset = numpy.random.choice(N, NUM_NEIGHBOR_POINT, replace=False)
			else:
				subset = list(range(N)) + list(numpy.random.choice(N, NUM_NEIGHBOR_POINT-N, replace=True))
			neighbor_points[i,:,:] = train_neighbor_points[points_idx][subset, :]
			input_add[i,:] = train_add[points_idx][subset]
			N = train_neighbor_count[points_idx] + train_inlier_count[points_idx]
			if N >= (NUM_NEIGHBOR_POINT+NUM_INLIER_POINT):
				subset = numpy.random.choice(N, NUM_NEIGHBOR_POINT+NUM_INLIER_POINT, replace=False)
			else:
				subset = list(range(N)) + list(numpy.random.choice(N, NUM_NEIGHBOR_POINT+NUM_INLIER_POINT-N, replace=True))
			input_motion[i,:] = train_motion[points_idx][subset]
		_, ls, ap, ar, rp, rr, mp, mr = sess.run([net.train_op, net.loss, net.add_prc, net.add_rcl, net.remove_prc, net.remove_rcl, net.motion_prc, net.motion_rcl],
			{net.inlier_pl:inlier_points, net.neighbor_pl:neighbor_points, net.add_mask_pl:input_add, net.remove_mask_pl:input_remove, net.motion_mask_pl:input_motion})
		loss_arr.append(ls)
		add_prc_arr.append(ap)
		add_rcl_arr.append(ar)
		rmv_prc_arr.append(rp)
		rmv_rcl_arr.append(rr)
		mot_prc_arr.append(mp)
		mot_rcl_arr.append(mr)
	epoch_time.append(time.time() - start_time)
	print("Epoch %d loss %.2f add %.2f/%.2f rmv %.2f/%.2f"%(epoch,numpy.mean(loss_arr),numpy.mean(add_prc_arr),numpy.mean(add_rcl_arr),numpy.mean(rmv_prc_arr), numpy.mean(rmv_rcl_arr)))

	if VAL_AREA is not None and epoch % VAL_STEP == VAL_STEP - 1:
		loss_arr = []
		add_prc_arr = []
		add_rcl_arr = []
		rmv_prc_arr = []
		rmv_rcl_arr = []
		mot_prc_arr = []
		mot_rcl_arr = []
		num_batches = int(len(val_inlier_points) / BATCH_SIZE)
		for batch_id in range(num_batches):
			start_idx = batch_id * BATCH_SIZE
			end_idx = (batch_id + 1) * BATCH_SIZE
			for i in range(BATCH_SIZE):
				points_idx = start_idx+i
				N = val_inlier_count[points_idx]
				if N >= NUM_INLIER_POINT:
					subset = numpy.random.choice(N, NUM_INLIER_POINT, replace=False)
				else:
					subset = list(range(N)) + list(numpy.random.choice(N, NUM_INLIER_POINT-N, replace=True))
				inlier_points[i,:,:] = val_inlier_points[points_idx][subset, :]
				input_remove[i,:] = val_remove[points_idx][subset]
				N = val_neighbor_count[points_idx]
				if N >= NUM_INLIER_POINT:
					subset = numpy.random.choice(N, NUM_NEIGHBOR_POINT, replace=False)
				else:
					subset = list(range(N)) + list(numpy.random.choice(N, NUM_NEIGHBOR_POINT-N, replace=True))
				neighbor_points[i,:,:] = val_neighbor_points[points_idx][subset, :]
				input_add[i,:] = val_add[points_idx][subset]
				N = val_neighbor_count[points_idx] + val_inlier_count[points_idx]
				if N >= (NUM_NEIGHBOR_POINT+NUM_INLIER_POINT):
					subset = numpy.random.choice(N, NUM_NEIGHBOR_POINT+NUM_INLIER_POINT, replace=False)
				else:
					subset = list(range(N)) + list(numpy.random.choice(N, NUM_NEIGHBOR_POINT+NUM_INLIER_POINT-N, replace=True))
				input_motion[i,:] = val_motion[points_idx][subset]
			ls, ap, ar, rp, rr, mp, mr = sess.run([net.loss, net.add_prc, net.add_rcl, net.remove_prc, net.remove_rcl, net.motion_prc, net.motion_rcl],
				{net.inlier_pl:inlier_points, net.neighbor_pl:neighbor_points, net.add_mask_pl:input_add, net.remove_mask_pl:input_remove, net.motion_mask_pl:input_motion})
			loss_arr.append(ls)
			add_prc_arr.append(ap)
			add_rcl_arr.append(ar)
			rmv_prc_arr.append(rp)
			rmv_rcl_arr.append(rr)
			mot_prc_arr.append(mp)
			mot_rcl_arr.append(mr)
		print("Validation %d loss %.2f add %.2f/%.2f rmv %.2f/%.2f"%(epoch,numpy.mean(loss_arr),numpy.mean(add_prc_arr),numpy.mean(add_rcl_arr),numpy.mean(rmv_prc_arr), numpy.mean(rmv_rcl_arr)))

print("Avg Epoch Time: %.3f" % numpy.mean(epoch_time))
# print("GPU Mem: " , tf.config.experimental.get_memory_info('GPU:0'))
saver.save(sess, MODEL_PATH)