import matplotlib.pyplot as plt
import rosbag
import argparse
import numpy as np
import pdb

def id_delays(bagfile):
	b = rosbag.Bag(bagfile)
	
	t_se = []; df_se = []; a_se = []
	t_a_long = []; a_long = []

	t_a_cmd = []; a_cmd = [];
	t_df_cmd = []; df_cmd = [];

	# Measured Acceleration and Steering
	for topic, msg, t in b.read_messages(topics='/vehicle/state_est'):
		t_se.append(msg.header.stamp.secs + 1e-9 * msg.header.stamp.nsecs)
		a_se.append(msg.a) # as of 7/20/18, filtered with alpha = 0.01, may be too aggressive
		df_se.append(msg.df) # this one should be fine, no filtering involved

	# Measured Longitudinal Acceleration (Vehicle IMU)
	for topic, msg, _ in b.read_messages(topics='/vehicle/imu'):
		t_a_long.append(msg.header.stamp.secs + 1e-9 * msg.header.stamp.nsecs)	
		a_long.append(msg.long_accel)
	
	# MPC Commands (Desired Values)	
	for topic, msg, t in b.read_messages(topics='/control/accel'):
		t_a_cmd.append(t.secs + 1e-9 * t.nsecs)
		a_cmd.append(msg.data)
	for topic, msg, t in b.read_messages(topics='/control/steer_angle'):
		t_df_cmd.append(t.secs + 1e-9 * t.nsecs)
		df_cmd.append(msg.data)

	# Messy way to estimate enable time (first optimal solution):
	# Assume first two commands are not solved to optimality, ignore those.
	t_enable = None

	count = 0
	t_accel = None
	for topic, msg, m_tm in b.read_messages(topics='/control/accel'):
		count = count + 1

		if count == 3:
			t_accel = m_tm.secs + m_tm.nsecs * 1e-9
			break

	count = 0
	t_steer = None
	for topic, msg, m_tm in b.read_messages(topics='/control/steer_angle'):
		count = count + 1

		if count == 3:
			t_steer = m_tm.secs + m_tm.nsecs * 1e-9
			break
	
	if t_accel == None and t_steer == None:
		t_enable = 0.0
	else:
		t_enable = np.max([t_accel, t_steer])

	t_se     = np.array(t_se)     - t_enable
	t_a_long = np.array(t_a_long) - t_enable
	t_a_cmd  = np.array(t_a_cmd)  - t_enable
	t_df_cmd = np.array(t_df_cmd) - t_enable

	df_se = [np.nan if t_se[x] < 0 else df_se[x] for x in range(len(t_se))]
	a_se  = [np.nan if t_se[x] < 0 else a_se[x] for x in range(len(t_se))]  
	a_long = [np.nan if t_a_long[x] < 0 else a_long[x] for x in range(len(t_a_long))]

	a_cmd = [np.nan if t_a_cmd[x] < 0 else a_cmd[x] for x in range(len(t_a_cmd))]
	df_cmd = [np.nan if t_df_cmd[x] < 0 else df_cmd[x] for x in range(len(t_df_cmd))]

	plt.figure()
	plt.subplot(211)
	plt.plot(t_a_cmd, a_cmd, 'r', label='MPC')

	###################################################################################
	for td_acc in [0.35, 0.4, 0.45]:		
		t_sampling = 0.1
		alpha_acc = t_sampling / (td_acc + t_sampling)		

		acc_filt = [0.0]

		for i in range(len(a_cmd)-1):
			if t_a_cmd[i] < 0:
				acc_filt.append(0.0)
			else:
				acc_next = alpha_acc * a_cmd[i] + (1 - alpha_acc) * acc_filt[-1]
				acc_filt.append(acc_next)
		

		plt.plot(t_a_cmd, acc_filt, '.', label='Delay='+str(td_acc))
	###################################################################################
	plt.plot(t_a_long[::10], a_long[::10], 'k', label='ACT')
	plt.xlabel('t (s)')
	plt.ylabel('Acceleration (m/s^2)')
	plt.legend()

	plt.subplot(212)
	plt.plot(t_df_cmd, df_cmd, 'r', label='MPC')
	###################################################################################
	for td_spas in [0.05, 0.1, 0.15]:
		t_sampling = 0.1
		alpha_spas = t_sampling / (td_spas + t_sampling)

		df_filt = [0.0]

		for i in range(len(df_cmd)-1):
			if t_df_cmd[i] < 0:
				df_filt.append(0.0)
			else:
				df_next = alpha_spas * df_cmd[i] + (1 - alpha_spas) * df_filt[-1]
				df_filt.append(df_next)

		plt.plot(t_df_cmd, df_filt, '.', label='Delay'+str(td_spas))
	###################################################################################
	plt.plot(t_se, df_se, 'k', label='ACT')
	plt.xlabel('t (s)')
	plt.ylabel('Steer Angle (rad)')
	plt.legend()

	plt.suptitle('Low Level Tracking Response')
	plt.show()

if __name__=='__main__':
	parser = argparse.ArgumentParser('IDs low level control delays.')
	parser.add_argument('--bf',  type=str, required=True, help='Bag file for path followed.')
	args = parser.parse_args()
	id_delays(args.bf)
