#!/usr/bin/env julia

# MPC Command Publisher/Controller Module Interface to the Genesis.
# This version using a Linearized Dynamic Bicycle Model w/ Gurobi - in progress.

###########################################
#### ROBOTOS
###########################################
using RobotOS
@rosimport genesis_path_follower.msg: state_est_dyn
@rosimport genesis_path_follower.msg: mpc_path_dyn
@rosimport std_msgs.msg: UInt8
@rosimport std_msgs.msg: Float32
rostypegen()
using genesis_path_follower.msg
using std_msgs.msg
using PyCall

###########################################
#### LOAD ROSPARAMS
###########################################
if has_param("mat_waypoints")
	mat_fname = get_param("mat_waypoints")
else
	error("No Matfile of waypoints provided!")
end

if has_param("track_using_time") && has_param("target_vel")
	track_with_time = get_param("track_using_time")
	target_vel = get_param("target_vel")	
else
	error("Invalid rosparam trajectory definition: track_using_time and target_vel")
end

if has_param("scripts_dir")
	scripts_dir = get_param("scripts_dir")
else
	error("Did not provide the scripts directory!")
end

if(has_param("lat0") && has_param("lat0") && has_param("lat0"))
	lat0 = get_param("lat0")
	lon0 = get_param("lon0")
	yaw0 = get_param("yaw0")
else
	error("Invalid rosparam global origin provided!")
end

###########################################
#### MPC Controller Module
#### Global Variables for Callbacks/Control Loop.
###########################################
push!(LOAD_PATH, scripts_dir * "mpc_utils")
import GPSDynMPCPathFollowerGurobi
const dmpc = GPSDynMPCPathFollowerGurobi
import DynBicycleModel
const dyn_mdl = DynBicycleModel

###########################################
#### Reference GPS Trajectory Module
###########################################
# Access Python modules for path processing.  Ugly way of doing it, can seek to clean this up in the future.
using PyCall
const gps_utils_loc = scripts_dir * "gps_utils"
unshift!(PyVector(pyimport("sys")["path"]), gps_utils_loc) # append the current directory to Python path
@pyimport ref_gps_traj_dyn as rgt
grt = rgt.GPSRefTrajectory(mat_filename=mat_fname, LAT0=lat0, LON0=lon0, YAW0=yaw0, traj_horizon = dmpc.N, traj_dt = dmpc.dt)

# Reference for MPC
const t_ref = collect(0:dmpc.dt:dmpc.N*dmpc.dt)
nx = 6; nu = 2; nH = length(t_ref)
x_ref = zeros(nH+1)		# first element is the point closest to the current state, essentially x0.
y_ref = zeros(nH+1)
psi_ref = zeros(nH+1)
vx_ref = zeros(nH+1)
vy_ref = zeros(nH+1)
wz_ref = zeros(nH+1)
A_lin  = zeros(nx,nx,nH)
B_lin  = zeros(nx,nu,nH)
g_lin  = zeros(nx,nH)

if target_vel > 0.0
	des_speed = target_vel
else
	des_speed = 0.00
end

ref_lock = false				
received_reference = false
x_curr  = 0.0
y_curr  = 0.0
psi_curr  = 0.0
v_curr  = 0.0
vx_curr  = 0.0
vy_curr  = 0.0
wz_curr  = 0.0
prev_a = 0.1
prev_df = 0.001
command_stop = false

###########################################
#### State Estimation Callback.
###########################################
function state_est_callback(msg::state_est_dyn)

	global x_curr, y_curr, psi_curr, v_curr, vx_curr, vy_curr, wz_curr
	global received_reference

	if ref_lock == false
		x_curr = msg.x
		y_curr = msg.y
		psi_curr = msg.psi
		v_curr = msg.v
		vx_curr = msg.vx
		vy_curr = msg.vy
		wz_curr = msg.wz
		received_reference = true
	end
end

###########################################
#### Linearization Model Update.
###########################################
function get_linearization(z0::Array{Float64,1}, u0::Array{Float64,1})
	As = zeros(nx,nx,nH)
	Bs = zeros(nx,nu,nH)
	gs = zeros(nx,nH)

	zs_nom = zeros(nx,nH+1)
	zs_nom[:,1] = z0

	zk = z0
	uk = u0
	for i = 1:nH
		#if i == 1
		#	lin_point = vec([ z0 ; u0])
		#else
		#	lin_point = vec([ zref[i,:] ; uref[i,:])
		#end
		lin_point = vec([ zk ; uk])	
		f = dyn_mdl.f_dyn_bicycle_model( lin_point  )
		J = dyn_mdl.jacobian_f_dyn_bicycle_model( lin_point )
		Al = J[:,1:6]
		Bl = J[:,7:8]

		As[:,:,i] = eye(nx) + J[:,1:6]*dmpc.dt
		Bs[:,:,i] = J[:,7:8]*dmpc.dt
		gs[:,i] = ( convert(Array{Float64,1}, f) - As[:,:,i]*lin_point[1:6] - Bs[:,:,i]*lin_point[7:8] )*dmpc.dt

		zk = convert(Array{Float64,1}, zk + f*dmpc.dt)	
		zs_nom[:,i+1] = zk	
	end

	# Test reconstructed output.
	zs_recon = zeros(nx,nH+1)
	zs_recon[:,1] = z0
	u_const = u0
	for i = 1:nH
		zs_recon[:,i+1] = As[:,:,i]*zs_recon[:,i] + Bs[:,:,i]*u_const + gs[:,i]
	end
		
	for i = 1:(nH+1)
		println(zs_recon[:,i] - zs_nom[:,i])
	end

	return As,Bs,gs
end

###########################################
#### Main Publishing Loop.
###########################################
function pub_loop(acc_pub_obj, steer_pub_obj, mpc_path_pub_obj)
    loop_rate = Rate(25.0)
	gc() # clear garbage
    while ! is_shutdown()
	    if ! received_reference		# Reference not received so don't use MPC yet.
	        rossleep(loop_rate)
	        continue
	    end

	    global ref_lock				# Ref lock used to ensure that get/set of state doesn't happen simultaneously.
	    ref_lock = true

		global x_curr, y_curr, psi_curr, v_curr, vx_curr, vy_curr, wz_curr, des_speed, command_stop

		if ! track_with_time		
			# fixed velocity-based path tracking
			x_ref, y_ref, psi_ref, vx_ref, vy_ref, wz_ref, stop_cmd = grt[:get_waypoints](x_curr, y_curr, psi_curr, des_speed)

			if stop_cmd == true
				command_stop = true
			end

		else
			# trajectory tracking
			x_ref, y_ref, psi_ref, vx_ref, vy_ref, wz_ref, stop_cmd = grt[:get_waypoints](x_curr, y_curr, psi_curr)

			if stop_cmd == true
				command_stop = true
			end
		end
		
		# Update Model
		z_ref = [x_ref y_ref psi_ref vx_ref vy_ref wz_ref]
		z_curr = vec([x_curr; y_curr; psi_curr; vx_curr; vy_curr; wz_curr])
		u_curr = vec([prev_a; prev_df])

		A_lin, B_lin, g_lin = get_linearization(z_curr, u_curr)
		
	    ref_lock = false
		if command_stop == false
			# disable garbage collection (makes optimization code run faster)
			gc_enable(false) # enable later on

			a_opt, df_opt, is_opt, solv_time, state_preds, input_preds = 
				dmpc.solve_gurobi(z_curr, u_curr, x_ref[2:end], y_ref[2:end], psi_ref[2:end], vx_ref[2:end], A_lin, B_lin, g_lin)

			rostm = get_rostime()
			tm_secs = rostm.secs + 1e-9 * rostm.nsecs

		    log_str = @sprintf("Solve Status: %s, Acc: %.3f, SA: %.3f, ST: %.3f", is_opt, a_opt, df_opt, solv_time)
		    println(log_str)

			publish( acc_pub_obj,   Float32Msg(a_opt) )
			publish( steer_pub_obj, Float32Msg(df_opt) )

			global prev_a, prev_df
			prev_a = a_opt
			prev_df = df_opt

			mpc_path_msg = mpc_path_dyn()
			mpc_path_msg.header.stamp = rostm
			mpc_path_msg.solv_status  = string(is_opt)
			mpc_path_msg.solv_time = solv_time

			mpc_path_msg.xs   = state_preds[:,1] 	# x_mpc
			mpc_path_msg.ys   = state_preds[:,2] 	# y_mpc
			mpc_path_msg.psis = state_preds[:,3] 	# psi_mpc
			mpc_path_msg.vxs  = state_preds[:,4] 	# vx_mpc
			mpc_path_msg.vys  = state_preds[:,5] 	# vy_mpc
			mpc_path_msg.wzs  = state_preds[:,6] 	# vz_mpc

			mpc_path_msg.xr   = z_ref[:,1] 	# x_ref
			mpc_path_msg.yr   = z_ref[:,2] 	# y_ref
			mpc_path_msg.psir = z_ref[:,3] 	# psi_ref
			mpc_path_msg.vxr  = z_ref[:,4] 	# vx_ref
			mpc_path_msg.vyr  = z_ref[:,5] 	# vy_ref
			mpc_path_msg.wzr  = z_ref[:,6] 	# vz_ref

			mpc_path_msg.df   = input_preds[:,2]	# d_f
			mpc_path_msg.acc  = input_preds[:,1]	# acc
			publish(mpc_path_pub_obj, mpc_path_msg)

			# re-enable garbage collection
			gc_enable(true)
		else
			publish( acc_pub_obj,   Float32Msg(-1.0) )
			publish( steer_pub_obj, Float32Msg(0.0) )		
		end
	    rossleep(loop_rate)
	end
end	

function start_mpc_node()
    init_node("dbw_mpc_pf")
    acc_pub   = Publisher("/control/accel", Float32Msg, queue_size=2)
    steer_pub = Publisher("/control/steer_angle", Float32Msg, queue_size=2)

    acc_enable_pub   = Publisher("/control/enable_accel", UInt8Msg, queue_size=2, latch=true)
    steer_enable_pub = Publisher("/control/enable_spas",  UInt8Msg, queue_size=2, latch=true)

    mpc_path_pub = Publisher("mpc_path_dyn", mpc_path_dyn, queue_size=2)
	sub_state  = Subscriber("state_est_dyn", state_est_dyn, state_est_callback, queue_size=2)    

	publish(acc_enable_pub, UInt8Msg(2))
	publish(steer_enable_pub, UInt8Msg(1))

    pub_loop(acc_pub, steer_pub, mpc_path_pub)    
end

if ! isinteractive()
	#try 
    	start_mpc_node()
    #catch x
    #	print(x)
    #end
end
