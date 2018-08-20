#***import matplotlib.pyplot as plt
import rosbag
import argparse
import numpy as np
import scipy.io as sio
import math
import pdb
import matplotlib.pyplot as plt

def sys_id_tire_model(bagfile, matfile):
	# Vehicle Parameters
	m = 2303.1 # kg, from HCE
	Iz = 5520.1 # kg * m^2, HCE
	lf = 1.5213 # m, from HCE
	lr = 1.4987 # m, from HCE
	C_alpha_f = 76419.5938188382
	C_alpha_r = 134851.89464029987
	# Cornering Stiffness front: 76419.5938188382  (HCE)
	# Cornering Stiffness rear: 134851.89464029987 (HCE)
	# width = 1.89 m, from HCE

	# Measurements
	tm  = [] 		#***
	vx = []  		#***
	vy = []
	wz = []
	df = []
	
	alpha_f = []
	alpha_r = []

	tmp_tm = []
	#acc = []
	vx_dot = []		#***
	vy_dot = []
	wz_dot = []

	# Load Data Here.
	if matfile == '':
		b = rosbag.Bag(bagfile)

		# Measured Acceleration and Steering
		for topic, msg, _ in b.read_messages(topics='/vehicle/state_est_dyn'):
			tm.append(msg.header.stamp.secs + 1e-9 * msg.header.stamp.nsecs)
			vx.append(msg.vx)
			vy.append(msg.vy)
			wz.append(msg.wz)
			df.append(msg.df)
			alpha_f.append( msg.df - np.arctan2(msg.vy + lf*msg.wz, msg.vx))
			alpha_r.append(        - np.arctan2(msg.vy - lr*msg.wz, msg.vx))		

		# Measured Longitudinal Acceleration (Vehicle IMU)
		for topic, msg, _ in b.read_messages(topics='/vehicle/imu'):
			tmp_tm.append(msg.header.stamp.secs + 1e-9 * msg.header.stamp.nsecs)	
			# this is wrong, fix this.
			vx_dot.append(msg.long_accel)
			vy_dot.append(msg.lat_accel)
			wz_dot.append(msg.yaw_rate)
			
		vx_dot = np.interp(tm, tmp_tm, vx_dot)
		vy_dot = np.interp(tm, tmp_tm, vy_dot)
		wz_dot = np.interp(tm, tmp_tm, wz_dot)
	else:
		raise ValueError("Not implemented yet!")
		d = sio.loadmat(matfile)
		tm = np.ravel(d['t'])
		vx = np.ravel(d['vx']) 
		vy = np.ravel(d['vy']) 
		wz = np.ravel(d['wz']) 
		df = np.ravel(d['df'])

		for i in range(len(df)):
			alpha_f.append( msg.df - math.atan2(msg.vy + lf*msg.wz, msg.vx))
			alpha_r.append(        - math.atan2(msg.vy - lr*msg.wz, msg.vx))
			
		vx_dot = np.ravel(d['a_long'])
		vy_dot = np.ravel(d['a_lat'])
		wz_dot = []

	vy_dot_est = []
	wz_dot_est = []
	for i in range(len(vy_dot)):
		vy_dot_est.append( 1/m *(C_alpha_f * alpha_f[i] * math.cos(df[i]) + C_alpha_r * alpha_r[i]) - wz[i] * vx[i] )
		wz_dot_est.append( 1/Iz *(lf*C_alpha_f * alpha_f[i] * math.cos(df[i]) - lr * C_alpha_r * alpha_r[i]) )

	plt.ion()
	fig = plt.figure()
	plt.subplot(611)		
	plt.plot(tm, vx)
	plt.ylabel('vx (m/s)')

	plt.subplot(612)		
	plt.plot(tm, vy)
	plt.ylabel('vy (m/s)')

	plt.subplot(613)		
	plt.plot(tm, wz)
	plt.ylabel('wz (m/s)')
	
	plt.subplot(614)		
	plt.plot(tm, alpha_f)
	plt.ylabel('af (rad)')

	plt.subplot(615)		
	plt.plot(tm, alpha_r)
	plt.ylabel('ar (rad)')

	plt.subplot(616)		
	plt.plot(tm, df)
	plt.ylabel('df (rad)')
	plt.show()

	plt.figure()
	plt.subplot(311)
	plt.plot(tm, vx_dot)
	plt.ylabel('vx_dot (m/s^2)')

	plt.subplot(312)
	plt.plot(tm, vy_dot, 'b')
	plt.plot(tm, vy_dot_est, 'r')
	plt.ylabel('vy_dot (m/s^2)')

	plt.subplot(313)
	plt.plot(tm, wz_dot)
	plt.plot(tm, wz_dot_est, 'r')
	plt.ylabel('wz_dot (rad/s^2)')
	plt.show()

	plt.pause(30)
	##############################################################################
	# Set up Data Matrices
	'''
	A = np.nan * np.ones( (2*len(vx_dot), 2) )
	b = np.nan * np.ones( (2*len(vx_dot), 1) )

	for i in range(0, len(vx_dot), 2):
		A[2*i,   :] = ...
		A[2*i+1, :] = ...
		b[2*i,   :] = ...
		b[2*i+1, :] = ...	

	# Solve the Problem
	x = np.linalg.lstsq(A,b)

	C_alpha_f = x[0]
	C_alpha_r = x[1]

	# Reconstruct predictions given fit.
	vx_rec = []
	vy_rec = []
	wz_rec = []

	# Loop over the model and integrate

	# Plot results: Actual vs. Predicted with Tire Model
	'''

if __name__=='__main__':
	parser = argparse.ArgumentParser('ID linear tire model from rosbag.')
	parser.add_argument('--bf', type=str, required=True, help='Bag file with dynamics info.')
	parser.add_argument('--mf', type=str, default='', required=False, help='Mat file with dynamics info' )
	args = parser.parse_args()
	sys_id_tire_model(args.bf, args.mf)

	