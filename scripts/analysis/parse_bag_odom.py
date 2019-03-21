import argparse
import scipy.io as sio
import rosbag
import numpy as np
import math
from tf.transformations import euler_from_quaternion
import matplotlib.pyplot as plt
#import append_state_est as ase
import pdb

# Code to convert rosbag into a matfile for further analysis.
# For more details, 
# https://github.com/MPC-Car/GenesisAutoware/blob/master/ros/src/sensing/drivers/oxford_gps_eth/src/node.cpp
# https://github.com/MPC-Car/GenesisAutoware/blob/master/ros/src/sensing/drivers/oxford_gps_eth/ncomman.pdf
# https://www.oxts.com/app/uploads/2018/02/rtman.pdf

# Extract gps fix (latitude, longitude, position error)
def parse_gps_fix(bag):
	tms = []; status = []; service = []; 
	lats = []; lons = []; alts = []; 
	acc_E = []; acc_N = []; acc_D = []

	for topic, msg, _ in bag.read_messages(topics='/gps/fix'):		
		# http://docs.ros.org/kinetic/api/sensor_msgs/html/msg/NavSatFix.html		
		
		if msg.position_covariance_type < 2: 
			# unanticipated behavior: no estimate of error available
			pdb.set_trace()

		tms.append(msg.header.stamp.secs + 1e-9*msg.header.stamp.nsecs)

		# http://docs.ros.org/kinetic/api/sensor_msgs/html/msg/NavSatStatus.html, basically status=2 for dGPS
		status.append(msg.status.status)	
		service.append(msg.status.service)
		
		lats.append(msg.latitude)  # decimal degrees
		lons.append(msg.longitude) # decimal degrees
		alts.append(msg.altitude)  # meters

		# [0] -> E^2, [4] -> N^2, [8] -> D^2 (E,N,D = east, north, down error in m)
		aE = math.sqrt(msg.position_covariance[0]) # m
		aN = math.sqrt(msg.position_covariance[4]) # m
		aD = math.sqrt(msg.position_covariance[8]) # m
		acc_E.append(aE)
		acc_N.append(aN)
		acc_D.append(aD)

	out_dict = {}
	out_dict['t'] = tms
	out_dict['status'] = status 
	out_dict['service'] = service
	out_dict['lats'] = lats
	out_dict['lons'] = lons
	out_dict['alts'] = alts
	out_dict['acc_E'] = acc_E
	out_dict['acc_N'] = acc_N
	out_dict['acc_D'] = acc_D
	return out_dict

def parse_gps_vel(bag):
	tms = []; velE = []; velN = []; velU = []; 
	acc_velE = []; acc_velN = []; acc_velU = [];

	for topic, msg, _ in bag.read_messages(topics='/gps/vel'):
		if np.sum(msg.twist.covariance) == 0.0:
			# this likely means the covariance was not updated
			# data shouldn't be trusted, skip to next measurement
			pdb.set_trace()

		tms.append(msg.header.stamp.secs + 1e-9*msg.header.stamp.nsecs)
		velE.append(msg.twist.twist.linear.x) # vE, m/s
		velN.append(msg.twist.twist.linear.y) # vN, m/s
		velU.append(msg.twist.twist.linear.z) # vU, m/s

		# [0] -> E^2, [7] -> N^2, [14] -> D^2 (E,N,D = east, north, down error in m/s)
		a_vE = math.sqrt(msg.twist.covariance[0])  # E vel error
		a_vN = math.sqrt(msg.twist.covariance[7])  # N vel error
		a_vU = math.sqrt(msg.twist.covariance[14]) # U vel error
		acc_velE.append(a_vE)
		acc_velN.append(a_vN)
		acc_velU.append(a_vU)
		
	out_dict = {}
	out_dict['t'] = tms
	out_dict['vel_E'] = velE
	out_dict['vel_N'] = velN
	out_dict['vel_U'] = velU
	out_dict['acc_velE'] = acc_velE
	out_dict['acc_velN'] = acc_velN	
	out_dict['acc_velU'] = acc_velU	
	return out_dict

def parse_imu_data(bag):	
	# https://www.oxts.com/app/uploads/2018/02/rtman.pdf, p 107 for Euler Angles format (heading multiplied by -1 in code)	
	# linear_acceleration: x (forward), y (right), z (up)
	# angular_velocity:  x (forward), y (right), z(up)
	# q -> roll, pitch, heading (- sign)

	tms = []; roll = []; pitch =  []; psi = []  	# rad
	acc_roll = []; acc_pitch = []; acc_psi = [];	# rad
	gyro_x = []; gyro_y = []; gyro_minusz = [];		# rad/s
	accel_x = []; accel_y = []; accel_minusz = []; 	# m/s^2

	for topic, msg, _ in bag.read_messages(topics='/imu/data'):

		if msg.orientation_covariance[0] >= 0.0174:
			# default value if the covariance is unknown (0.0174532925)
			# should skip these results
			pdb.set_trace()

		tms.append(msg.header.stamp.secs + 1e-9*msg.header.stamp.nsecs)
		
		ori = msg.orientation
		quat = (ori.x, ori.y, ori.z, ori.w)
		r, p, y = euler_from_quaternion(quat)

		# Note: this is a bit of a hacky way to get what we want: psi = angle ccw from E = 0.
		# heading in the Genesis OxTS coord system is cw wrt N = 0 (longitudinal axis of vehicle).
		# in the OxTS driver code, there is a minus sign for heading
		# (https://github.com/MPC-Car/GenesisAutoware/blob/master/ros/src/sensing/drivers/oxford_gps_eth/src/node.cpp#L10)
		# so yaw from euler_from_quaternion output is actually ccw radians from N = 0.
		assert(-math.pi <= y)
		assert(math.pi >= y)

		y = y + 0.5 * math.pi
		if y > math.pi:
			y = - (2*math.pi - y)		

		roll.append(r) 	# rad
		pitch.append(p) # rad
		psi.append(y) 	# rad

		acr = math.sqrt(msg.orientation_covariance[0]) # roll accuracy (rad)
		acp = math.sqrt(msg.orientation_covariance[4]) # pitch accuracy (rad)
		acy = math.sqrt(msg.orientation_covariance[8]) # heading accuracy (rad)
		acc_roll.append(acr) 
		acc_pitch.append(acp)
		acc_psi.append(acy)

		gyro_x.append(msg.angular_velocity.x) 		# vehicle longitudinal axis (x=forward), rad/s
		gyro_y.append(msg.angular_velocity.y)		# vehicle lateral axis (y=right), rad/s
		gyro_minusz.append(msg.angular_velocity.z)	# vehicle up axis (-z), rad/s

		accel_x.append(msg.linear_acceleration.x)		# longitudinal acceleration (x = forward), m/s^2
		accel_y.append(msg.linear_acceleration.y)		# lateral acceleration (y = right), m/s^2
		accel_minusz.append(msg.linear_acceleration.z)	# vertical acceleration (-z = up), m/s^2
		
	out_dict = {}
	out_dict['t'] = tms	
	out_dict['roll'] = roll
	out_dict['pitch'] = pitch
	out_dict['psi'] = psi
	out_dict['acc_roll'] = acc_roll	
	out_dict['acc_pitch'] = acc_pitch
	out_dict['acc_psi'] = acc_psi
	out_dict['gyro_x'] = gyro_x
	out_dict['gyro_y'] = gyro_y
	out_dict['gyro_minusz'] = gyro_minusz
	out_dict['accel_x'] = accel_x
	out_dict['accel_y'] = accel_y
	out_dict['accel_minusz'] = accel_minusz
	out_dict['acc_ang_vel'] = math.sqrt(0.000436332313) # not time-varying, rad/s
	out_dict['acc_lin_accel'] = math.sqrt(0.0004)		# not time-varying, m/s^2
	return out_dict

def parse_ukf_odometry(bag):	
	# https://www.oxts.com/app/uploads/2018/02/rtman.pdf, p 107 for Euler Angles format (heading multiplied by -1 in code)	
	# linear_acceleration: x (forward), y (right), z (up)
	# angular_velocity:  x (forward), y (right), z(up)
	# q -> roll, pitch, heading (- sign)

	tms = []; 

	xs  = []; ys  = []; zs  = []; 
	acc_x = []; acc_y = []; acc_z = []
	
	roll = []; pitch =  []; psi = []  	# rad
	acc_roll = []; acc_pitch = []; acc_psi = [];	# rad
		
	vxs = []; vys = []; vzs = [];
	acc_vx = []; acc_vy = []; acc_vz = []
	wxs = []; wys = []; wzs = []
	acc_wx = []; acc_wy = []; acc_wz = []

	for topic, msg, _ in bag.read_messages(topics='/odometry/filtered'):				
		tms.append(msg.header.stamp.secs + 1e-9*msg.header.stamp.nsecs)

		pose_variance = np.diag( np.array(msg.pose.covariance).reshape(6,6) )
		xs.append( msg.pose.pose.position.x )
		ys.append( msg.pose.pose.position.y )
		zs.append( msg.pose.pose.position.z )
		acc_x.append( math.sqrt(pose_variance[0]) )
		acc_y.append( math.sqrt(pose_variance[1]) )
		acc_z.append( math.sqrt(pose_variance[2]) )

		ori = msg.pose.pose.orientation
		quat = (ori.x, ori.y, ori.z, ori.w)
		r, p, y = euler_from_quaternion(quat)
		roll.append(r) 	# rad
		pitch.append(p) # rad
		psi.append(y) 	# rad			
		acr = math.sqrt(pose_variance[3])
		acp = math.sqrt(pose_variance[4])
		acy = math.sqrt(pose_variance[5])		
		acc_roll.append(acr) 
		acc_pitch.append(acp)
		acc_psi.append(acy)

		vxs.append(msg.twist.twist.linear.x)
		vys.append(msg.twist.twist.linear.y)
		vzs.append(msg.twist.twist.linear.z)
		wxs.append(msg.twist.twist.angular.x)
		wys.append(msg.twist.twist.angular.y)
		wzs.append(msg.twist.twist.angular.z)		
		twist_variance = np.diag(np.array(msg.twist.covariance).reshape(6,6))
		acc_vx.append(math.sqrt(twist_variance[0]))
		acc_vy.append(math.sqrt(twist_variance[1]))
		acc_vz.append(math.sqrt(twist_variance[2]))
		acc_wx.append(math.sqrt(twist_variance[3]))
		acc_wy.append(math.sqrt(twist_variance[4]))
		acc_wz.append(math.sqrt(twist_variance[5]))

	out_dict = {}
	out_dict['t'] = tms	
	out_dict['roll'] = roll
	out_dict['pitch'] = pitch
	out_dict['psi'] = psi
	out_dict['acc_roll'] = acc_roll	
	out_dict['acc_pitch'] = acc_pitch
	out_dict['acc_psi'] = acc_psi	
	out_dict['x'] = xs
	out_dict['y'] = ys
	out_dict['z'] = zs
	out_dict['acc_x'] = acc_x
	out_dict['acc_y'] = acc_y
	out_dict['acc_z'] = acc_z	
	out_dict['vx'] = vxs
	out_dict['vy'] = vys
	out_dict['vz'] = vzs
	out_dict['acc_vx'] = acc_vx
	out_dict['acc_vy'] = acc_vy
	out_dict['acc_vz'] = acc_vz	
	out_dict['wx'] = wxs
	out_dict['wy'] = wys
	out_dict['wz'] = wzs
	out_dict['acc_wx'] = acc_wx
	out_dict['acc_wy'] = acc_wy
	out_dict['acc_wz'] = acc_wz	
	return out_dict	
	
def parse_mando(bag):	
	def parse_lane(bag, topic_name):	
		tms = []; poly_coeffs = [];
		mark_type = []; mark_quality = [];	
		for topic, msg, _ in bag.read_messages(topics=topic_name):
			tms.append(msg.header.stamp.secs + 1e-9*msg.header.stamp.nsecs)
			# a0, a1, a2, a3, y = sum_i a_i * x^i
			poly_coeffs.append([msg.a0, msg.a1, msg.a2, msg.a3])
			# lane_mark_type: 6 "Invalid" 5 "Botts'Dots" 4 "DoubleLaneMark" 3 "RoadEdge" 2 "Undecided" 1 "Solid" 0 "Dashed"	
			mark_type.append(msg.lane_mark_type)
			# lane_mark_quality: 0, 1 "Low Quality" 2, 3 "High Quality	
			mark_quality.append(msg.lane_mark_quality)

		res = {}
		res['t'] = tms
		res['poly_coeffs'] = poly_coeffs
		res['mark_type'] = mark_type
		res['mark_quality'] = mark_quality

		return res

	left_dict = parse_lane(bag, '/mando_camera/lane_left')
	right_dict = parse_lane(bag, '/mando_camera/lane_right')

	out_dict = {}
	out_dict['lane_left'] = left_dict
	out_dict['lane_right'] = right_dict
	return out_dict										

'''
def parse_image_timestamps(bag, topic_name='/out/compressed'):			
	# In progress
	# Approach 1: CvBridge
	#bridge = CvBridge()
	#cv2_img = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

	# Approach 2: imdecode	
	# Error: keep getting grayscale version when using imshow...
	out_dict = {}
	out_dict['t'] = []
	out_dict['seq'] = []
	#out_dict['img'] = []
	count = 0
	for topic, msg, t in bag.read_messages(topics = topic_name):
		#if (count % 3) > 0: # 30 Hz -> 10 Hz
		#	continue

		out_dict['t'].append(msg.header.stamp.secs + 1e-9*msg.header.stamp.nsecs)
		out_dict['seq'].append(msg.header.seq)

		#np_arr = np.fromstring(msg.data, np.uint8)		
		#out_dict['img'].append(cv2.imdecode(np_arr, cv2.IMREAD_COLOR))	
		count = count + 1		
	return out_dict
'''
def parse_driver_brake_override(bag):
	def parse_binary(bag, topic_name):
		tms = []; flag = [];
		for topic, msg, t in bag.read_messages(topics=topic_name):
			tms.append(t.secs + 1e-9 * t.nsecs)
			flag.append(msg.data)

		return np.array(tms), np.array(flag)

	tms_brake, flag_brake = parse_binary(bag, '/vehicle/driver_braking')
	tms_override, flag_override = parse_binary(bag, '/vehicle/driver_override')

	# time synchronization to combine everything into measurements relative to tms_brake
	flag_tm_synched_override = []
	for i in range(len(tms_brake)):
		tm_query = tms_brake[i]
		closest_ind_override = np.argmin( np.fabs(tms_override - tm_query) )
		flag_tm_synched_override.append( flag_override[closest_ind_override] )

	out_dict = {}
	# Essentially can assume (flag_override ^ not flag_brake = accelerating, flag_override ^ flag_brake = braking).
	out_dict['t'] = tms_brake
	out_dict['flag_brake'] = flag_brake
	out_dict['flag_override'] = flag_tm_synched_override
	return out_dict

def parse_vehicle_imu(bag):
	tms = []; yaw_rate = []; lat_accel = []; long_accel = []
	for topic, msg, t in bag.read_messages(topics='/vehicle/imu'):
		tms.append(msg.header.stamp.secs + 1e-9*msg.header.stamp.nsecs)
		yaw_rate.append(msg.yaw_rate)
		lat_accel.append(msg.lat_accel)
		long_accel.append(msg.long_accel)
	
	# should confirm signal quality/bias + lat accel direction
	out_dict = {}
	out_dict['t'] = tms
	out_dict['yaw_rate'] = yaw_rate
	out_dict['lat_accel'] = lat_accel
	out_dict['long_accel'] = long_accel
	return out_dict

def parse_vehicle_steering(bag):
	tms = []; swa = []; df = []
	for topic, msg, t in bag.read_messages(topics='/vehicle/steering'):
		tms.append(msg.header.stamp.secs + 1e-9 * msg.header.stamp.nsecs)
		swa.append(msg.steering_wheel_angle) # degrees, steering wheel angle
		df.append(math.radians(msg.steering_wheel_angle) / 15.87) # steering angle (tire), rad
	out_dict = {}
	out_dict['t'] = tms
	out_dict['swa'] = swa
	out_dict['df']  = df
	return out_dict
	
def parse_vehicle_wheel_speeds(bag):
	tms = []; ws_fl = []; ws_fr = []; ws_rl = []; ws_rr = []
	for topic, msg, t in bag.read_messages(topics='/vehicle/wheel_speeds'):
		tms.append(msg.header.stamp.secs + 1e-9 * msg.header.stamp.nsecs)
		# each measurement converted from kmph to m/s (1 m/s = 3.6 kmph)
		ws_fl.append(msg.wheel_speed_fl / 3.6) # front left wheel speed, m/s
		ws_fr.append(msg.wheel_speed_fr / 3.6) # front right wheel speed, m/s
		ws_rl.append(msg.wheel_speed_rl / 3.6) # rear left wheel speed, m/s
		ws_rr.append(msg.wheel_speed_rr / 3.6) # rear right wheel speed, m/s			
	out_dict = {}
	out_dict['t'] = tms
	out_dict['ws_fl'] = ws_fl
	out_dict['ws_fr'] = ws_fr
	out_dict['ws_rl'] = ws_rl
	out_dict['ws_rr'] = ws_rr
	return out_dict
	
def parse_odom(bag, res_dict):
	tms = []; xs = []; ys = []; zs = []; quats = []; psis = []
	
	for topic, msg, t in bag.read_messages(topics='/odom'):
		tms.append(msg.header.stamp.secs + 1e-9 * msg.header.stamp.nsecs)
		xs.append(msg.pose.pose.position.x)
		ys.append(msg.pose.pose.position.y)
		zs.append(msg.pose.pose.position.z)
		
		ori = msg.pose.pose.orientation
		q = (ori.x, ori.y, ori.z, ori.w)
		quats.append(q)
		r, p, y = euler_from_quaternion(q)
		psis.append(y + np.pi/2.0)			
					
	
	res_dict['t'] = tms
	res_dict['x'] = xs
	res_dict['y'] = ys
	res_dict['z'] = zs
	res_dict['quat'] = quats
	res_dict['psi'] = psis	
		
def parse_rosbag(in_rosbag, out_mat):	
	b = rosbag.Bag(in_rosbag)
	
	res_dict = {}	
	res_dict['src_bag'] = in_rosbag
	#res_dict['image'] = parse_image_timestamps(b)
	res_dict['gps_fix'] = parse_gps_fix(b)
	res_dict['gps_vel'] = parse_gps_vel(b)
	res_dict['imu_data'] = parse_imu_data(b)
	res_dict['lane_detection'] = parse_mando(b)
	res_dict['brake_override'] = parse_driver_brake_override(b)
	res_dict['vehicle_imu'] = parse_vehicle_imu(b)	
	res_dict['steering'] = parse_vehicle_steering(b)
	res_dict['wheel_speeds'] = parse_vehicle_wheel_speeds(b)
	parse_odom(b, res_dict)
	#res_dict['ukf_odom'] = parse_ukf_odometry(b)
	# get the local kinematic state information:
	#res_dict = ase.add_state_est(res_dict, 37.917929, -122.331798, from_matfile=False)
	# TODO: can consider using UKF odometry results instead for state_est.
	sio.savemat(out_mat, res_dict)
	
	# Useful time synch interpolation function (may need this later):
	# data_interp  = np.interp(t_interp, tm_raw, data_raw)				
	
if __name__=='__main__':
	parser = argparse.ArgumentParser('Plot processed matfile containing state/input history from a path following experiment.')
	parser.add_argument('-i', '--infile',  type=str, required=True, help='Input: Bag File.')
	parser.add_argument('-o', '--outfile', type=str, required=True, help='Output: Mat File.')
	args = parser.parse_args()
	parse_rosbag(args.infile, args.outfile)
