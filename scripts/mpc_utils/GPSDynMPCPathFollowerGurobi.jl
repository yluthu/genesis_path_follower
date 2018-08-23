#!/usr/bin/env julia

#=
 This is parser for solving a linearized dynamic bicycle model path following problem.

 Licensing Information: You are free to use or extend these projects for 
 education or research purposes provided that (1) you retain this notice
 and (2) you provide clear attribution to UC Berkeley, including a link 
 to http://barc-project.com

 Attibution Information: The barc project ROS code-base was developed
 at UC Berkeley in the Model Predictive Control (MPC) lab by Jon Gonzales
 (jon.gonzales@berkeley.edu). The cloud services integation with ROS was developed
 by Kiet Lam  (kiet.lam@berkeley.edu). The web-server app Dator was 
 based on an open source project by Bruce Wootton
=# 

module GPSDynMPCPathFollowerGurobi
	__precompile__()
	using Gurobi
	println("Creating linearized dynamic bicycle model in Gurobi ....")

	# ====================== general problem formulation is given by ======================
	# x_{k+1} = A_k x_k + B_k u_k + g_k
	# u_lb <= u_k <= u_ub
	# x_lb <= x_k <= x_ub
	# dU_lb <= u_k - u_{k-1} <= dU_ub     (can be set to a large number of desired)
	# minimize (x_k - x_k_ref)' * Q * (x_k - x_k_ref) + (u_k - u_k_ref)' * R * (u_k - u_k_ref) + (u_k - u_{k-1})' * Rdelta * (u_k - u_{k-1}) 
	#
	# if no references available, simply put weight in Q/R matrix =0
	#
	# x = (X, Y, psi, vx, vy, wz), u = (a, df)


	# ================== Phase 0 (define and build up everything) ======================

	# general control parameters(not all might be needed)
	dt 		= 0.2
	N 		= 8
	nx 		= 6				# dimension of x = (X, Y, psi, vx, vy, wz)
	nu 		= 2				# number of inputs u = (a, df)
	L_a 	= 1.5213		# from CoG to front axle (according to Jongsang)
	L_b 	= 1.4987		# from CoG to rear axle (according to Jongsang)

	# system dynamics A, B, c
	# these are defined just to solve a first optimization problem (they will be updated later on)
	A_init = zeros(nx, nx, N)
	B_init = zeros(nx, nu, N)
	g_init = zeros(nx, N)
	for i = 1 : N
		A_init[:,:,i] = eye(nx) 						# randomly generated for 1st solve
		B_init[:,:,i] = [eye(nu) ; eye(nu) ; eye(nu)]	# ditto
		g_init[:,i] = zeros(nx)				# ditto 
	end

	# define cost functions (need to define them all) ==> @VG: double check, DONE!.
	# states
	C_X 	= 9.0
	C_Y 	= 9.0
	C_psi 	= 10.0
	C_vx 	= 0   		# don't want to track reference velocity?
	C_vy 	= 0
	C_wz 	= 0

	# cost on inputs
	C_a 	= 0
	C_df 	= 0

	# cost on Delta-inputs
	C_da 	= 100
	C_ddf 	= 1000

	# forming now the standard Q, R, Rdelta matrices
	Q = diagm([C_X ; C_Y ; C_psi ; C_vx ; C_vy ; C_wz])	# state cost
	R = diagm([C_a ; C_df]) 							# input cost
	Rdelta = diagm([C_da ; C_ddf]) 						# deltaU cost

	# define (box) constraints on ALL states, inputs, and Delta-input
	# if some variables are "unbounded", just bound them with a largeNumber (make sure this number is large enough)
	largeNumber = 5e3			# use this number for variables that are not upper/lower bounded
	
	# state constraints
	X_max 		= largeNumber; 	X_min	= -largeNumber
	Y_max		= largeNumber;	Y_min 	= -largeNumber 
	psi_max		= largeNumber;	psi_min	= -largeNumber
	vx_max		= 30;			vx_min 	= 0
	vy_max 		= largeNumber;	vy_min	= -largeNumber
	wz_max		= largeNumber;	wz_min	= -largeNumber

	# input constraints
	a_max		= 2;			a_min	= -3
	df_max 		= 0.4;			df_min 	= -0.4

	# delta-U constraints
	da_max		= 1.5;			da_min	= -1.5
	ddf_max		= 0.4;			ddf_min	= -0.4

	## build state constraints
	x_lb = [	X_min
				Y_min
				psi_min
				vx_min
				vy_min
				wz_min		]

	x_ub = [	X_max
				Y_max
				psi_max
				vx_max
				vy_max
				wz_max		]

	## build input constraints	
	u_lb = [	a_min
				df_min 	]
	u_ub = [ 	a_max
				df_max 	]

	## build delta-U constraints
	# take into account time differentiation
	dU_lb = [	da_min
				ddf_min 	]
	dU_ub = [ 	da_max
				ddf_max 	]

	# build references; need references to ALL states and inputs for 1st solve
	# if reference not available/needed, put zeros and set weight in cost to zero
	# the references contain DO NOT containt the current state (hence, of length N = prediction horizon)
	X_ref_init 		= zeros(N)
	Y_ref_init 		= zeros(N)
	psi_ref_init 	= zeros(N)
	vx_ref_init		= zeros(N)
	vy_ref_init 	= zeros(N)
	wz_ref_init 	= zeros(N)

	x_ref_init = zeros(N*nx)
	for i = 1 : N
		x_ref_init[(i-1)*nx+1] = X_ref_init[i]		# set x_ref
		x_ref_init[(i-1)*nx+2] = Y_ref_init[i]		# set v_ref
		x_ref_init[(i-1)*nx+3] = psi_ref_init[i]	# set v_ref
		x_ref_init[(i-1)*nx+4] = vx_ref_init[i]		# set v_ref
		x_ref_init[(i-1)*nx+5] = vy_ref_init[i]		# set v_ref
		x_ref_init[(i-1)*nx+6] = wz_ref_init[i]		# set v_ref
	end	

	# input reference
	a_ref_init 		= zeros(N)
	df_ref_init 	= zeros(N)

	u_ref_init = zeros(N*nu)	# if not used, set cost to zeros
	for i = 1 : N
		u_ref_init[(i-1)*nu+1] = a_ref_init[i]		# set x_ref
		u_ref_init[(i-1)*nu+2] = df_ref_init[i]		# set v_ref
	end	

	# get Initial state and input
	# should be done dynamically later on
	X0_init 	= 0
	Y0_init 	= 0
	psi0_init	= 0
	vx0_init 	= 0
	vy0_init	= 0
	wz0_init	= 0

	a0_init		= 0
	df0_init 	= 0

	x0_init = [X0_init ; Y0_init ; psi0_init ; vx0_init ; vy0_init ; wz0_init]
	u0_init = [a0_init ; df0_init]

	# ================== Transformation 1 (everything automated) ======================
	# augment state and redefine system dynamics (A,B,g) and constraints
	# x_tilde_k := (x_k , u_{k-1})
	# u_tilde_k := (u_k - u_{k-1})
	A_tilde_init = zeros(nx+nu,nx+nu,N)
	B_tilde_init = zeros(nx+nu,nu,N)
	g_tilde_init = zeros(nx+nu,N)
	for i = 1 : N 
		A_tilde_init[:,:,i] = [ A_init[:,:,i]  		B_init[:,:,i] 
								zeros(nu,nx)   		eye(nu)			]
		B_tilde_init[:,:,i] = [	B_init[:,:,i] 	;  	eye(nu)	]
		g_tilde_init[:,i] =   [	g_init[:,i]		; 	zeros(nu) ]
	end

	x_tilde_lb = [x_lb ; u_lb]
	x_tilde_ub = [x_ub ; u_ub]
	u_tilde_lb = dU_lb
	u_tilde_ub = dU_ub

	Q_tilde = [	Q 				zeros(nx,nu)
				zeros(nu,nx)	R 				]

	R_tilde = Rdelta 

	x_tilde_0_init = [x0_init ; u0_init]	# initial state of augmented system

	x_tilde_ref_init = zeros(N*(nx+nu))
	for i = 1 : N
		x_tilde_ref_init[(i-1)*(nx+nu)+1 : (i-1)*(nx+nu)+nx] = x_ref_init[(i-1)*nx+1 : i*nx]
		x_tilde_ref_init[(i-1)*(nx+nu)+nx+1 : (i-1)*(nx+nu)+nx+nu] = u_ref_init[i]
	end

	u_tilde_ref_init = zeros(N*nu) 	# goal is to minimize uTilde = (u_k - u_{k-1})

	# ================== Transformation 2 (bring into standard QP form) ======================
	# bring into GUROBI format
	# minimize_z    z' * H * z + f' * z    (note that Gurobi requires 1/2*z*(2*H)*z + f*z)
	#	s.t.		A_eq * z = b_eq
	#				A * z <= b
	#				z_lb <= z <= z_ub

	# z := (u_tilde_0, x_tilde_1 , u_tilde_1 x_tilde_2 , ... u_tilde_{N-1}, x_tilde_N , )
	
	n_uxu = nu+nx+nu 	# size of one block of (u_tilde, x_tilde) = (deltaU, x, u)

	# Build cost function
	# cost for (u_tilde, x_tilde) = (deltaU , S, V, U)
	H_block = [	R_tilde 			zeros(nu, nu+nx)
				zeros(nu+nx,nu) 	Q_tilde				]
	H_gurobi = kron(eye(N), H_block)

	z_gurobi_ref_init = zeros(N*n_uxu) 	# reference point for z_gurobi ; PARAMETER!
	for i = 1 : N
		z_gurobi_ref_init[(i-1)*n_uxu+1 : (i-1)*n_uxu+nu] = u_tilde_ref_init[(i-1)*nu+1 : i*nu] 		# should be zero for this application
		z_gurobi_ref_init[(i-1)*n_uxu+nu+1 : i*n_uxu] = x_tilde_ref_init[(i-1)*(nx+nu)+1 : i*(nu+nx)]
	end 

	f_gurobi_init = -2*H_gurobi*z_gurobi_ref_init

	# build box constraints lb_gurobi <= z <= ub_gurobi
	# recall: z = (u_tilde, x_tilde, ....)
	lb_gurobi = repmat([u_tilde_lb ; x_tilde_lb], N)		# (deltaU, X, U)
	ub_gurobi = repmat([u_tilde_ub ; x_tilde_ub], N)		# (deltaU, X, U)

	# build equality matrix (most MALAKA task ever)
	nu_tilde = nu
	nx_tilde = nu+nx
	# n_uxu = nu_tilde + nx_tilde
	Aeq_gurobi_init = zeros(N*nx_tilde , N*(nx_tilde+nu_tilde))
	Aeq_gurobi_init[1:nx_tilde, 1:(nx_tilde+nu_tilde)] = [-B_tilde_init[:,:,1] eye(nx_tilde)] 	# fill out first row associated with x_tilde_1
	for i = 2 : N  	# fill out rows associated to x_tilde_2, ... , x_tilde_N
		Aeq_gurobi_init[ (i-1)*nx_tilde+1 : i*nx_tilde  , (i-2)*(nu_tilde+nx_tilde)+(nu_tilde)+1 : (i-2)*(nu_tilde+nx_tilde)+nu_tilde+(nx_tilde+nu_tilde+nx_tilde)    ] = [-A_tilde_init[:,:,i] -B_tilde_init[:,:,i] eye(nx_tilde)]
	end

	# right-hand-size of equality constraint
	beq_gurobi_init = zeros(N*nx_tilde)
	for i = 1 : N
		beq_gurobi_init[(i-1)*nx_tilde+1:i*nx_tilde] = g_tilde_init[:,i]
	end
	beq_gurobi_init[1:nx_tilde] = beq_gurobi_init[1:nx_tilde] + A_tilde_init[:,:,1]*x_tilde_0_init 	# PARAMETER: depends on x0

	# ================ Solve Problem =================== 
    tic()
    GurobiEnv = Gurobi.Env()
	setparam!(GurobiEnv, "Presolve", -1)	# -1 is automatic; 0 is off
	setparam!(GurobiEnv, "LogToConsole", 0)	# not output

	# add A = A_gurobi and b=b_gurobi for inequality constraint
	# note that: (1/2) * z' * H * z + f' * z
    GurobiModel = gurobi_model(GurobiEnv;
    			name = "qp_01",
    			H = 2*H_gurobi,
    			f = f_gurobi_init,	# PARAMETER that depends on x_ref and u_ref
    			Aeq = Aeq_gurobi_init,
    			beq = beq_gurobi_init,	# PARAMETER that depends on x0, u_{-1}	
    			lb = lb_gurobi,
    			ub = ub_gurobi		)
    optimize(GurobiModel)
	solv_time=toq()
	println("1st solv time Gurobi:  $(solv_time*1000) ms")

	## access results
	# sol = get_solution(GurobiModel)	# get optimizer
	# println("soln = $(sol)")
	# objv = get_objval(GurobiModel)	# get optimal value
	# println("objv = $(objv)")


	# this function is called iteratively
	# all parameters that change are passed on as arguments
	# NOTE: reference signals are of length N and should start at x1, x2, ..., xN
	#       format of A_updated, B_updated, g_updated should be same as above
	function solve_gurobi(x0::Array{Float64,1}, u0::Array{Float64,1}, X_ref::Array{Float64,1}, Y_ref::Array{Float64,1}, psi_ref::Array{Float64,1} ,vx_ref::Array{Float64,1}, A_updated::Array{Float64,3}, B_updated::Array{Float64,3}, g_updated::Array{Float64,2}   )
		# assumptions:
		#	x0 = [X0 ; Y0 ; psi0 ; vx0 ; vy0 ; wz0]   (current state)
		#	u0 = [a0 ; df0_init]  	(previous input)

		tic() 	# start timing

		# build tilde system problem
		x_tilde_0 = [x0 ; u0]	# initial state of system; PARAMETER

		# build x_ref_updated
		x_ref_updated = zeros(N*nx)		# most states are irrelevant or regulated to zero
		# only update the relevant states X_ref and Y_ref
		for i = 1 : N
			x_ref_updated[(i-1)*nx+1] = X_ref[i]		# set X_ref
			x_ref_updated[(i-1)*nx+2] = Y_ref[i]		# set Y_ref
			x_ref_updated[(i-1)*nx+3] = psi_ref[i]		# set psi_ref
			x_ref_updated[(i-1)*nx+4] = vx_ref[i]		# set v_ref
		end	


		# x_tilde transformation
		# update system matrices for tilde-notation
		A_tilde_updated = zeros(nx+nu,nx+nu,N)
		B_tilde_updated = zeros(nx+nu,nu,N)
		g_tilde_updated = zeros(nx+nu,N)
		for i = 1 : N 
			A_tilde_updated[:,:,i] = [ 	A_updated[:,:,i]  		B_updated[:,:,i] 
										zeros(nu,nx)   			eye(nu)			]
			B_tilde_updated[:,:,i] = [	B_updated[:,:,i] 	;  	eye(nu)	]
			g_tilde_updated[:,i] =   [	g_updated[:,i]		; 	zeros(nu) ]
		end

		x_tilde_ref_updated = zeros(N*(nx+nu))
		for i = 1 : N
			x_tilde_ref_updated[(i-1)*(nx+nu)+1 : (i-1)*(nx+nu)+nx] = x_ref_updated[(i-1)*nx+1 : i*nx]
			x_tilde_ref_updated[(i-1)*(nx+nu)+nx+1 : (i-1)*(nx+nu)+nx+nu] = u_ref_init[i] # did not change
		end

		# bring into z-format
		# z := (u_tilde_0, x_tilde_1 , u_tilde_1 x_tilde_2 , ... u_tilde_{N-1}, x_tilde_N , )
	
		z_gurobi_ref_updated = zeros(N*n_uxu) 	# reference point for z_gurobi ; PARAMETER!
		for i = 1 : N
			z_gurobi_ref_updated[(i-1)*n_uxu+1 : (i-1)*n_uxu+nu] = u_tilde_ref_init[(i-1)*nu+1 : i*nu] 		# no change
			z_gurobi_ref_updated[(i-1)*n_uxu+nu+1 : i*n_uxu] = x_tilde_ref_updated[(i-1)*(nx+nu)+1 : i*(nu+nx)]
		end 

		f_gurobi_updated = -2*H_gurobi*z_gurobi_ref_updated

		# z-transformation
		Aeq_gurobi_updated = zeros(N*nx_tilde , N*(nx_tilde+nu_tilde))
		Aeq_gurobi_updated[1:nx_tilde, 1:(nx_tilde+nu_tilde)] = [-B_tilde_updated[:,:,1] eye(nx_tilde)] 	# fill out first row associated with x_tilde_1
		for i = 2 : N  	# fill out rows associated to x_tilde_2, ... , x_tilde_N
			Aeq_gurobi_updated[ (i-1)*nx_tilde+1 : i*nx_tilde  , (i-2)*(nu_tilde+nx_tilde)+(nu_tilde)+1 : (i-2)*(nu_tilde+nx_tilde)+nu_tilde+(nx_tilde+nu_tilde+nx_tilde)    ] = [-A_tilde_updated[:,:,i] -B_tilde_updated[:,:,i] eye(nx_tilde)]
		end

		# right-hand-size of equality constraint
		beq_gurobi_updated = zeros(N*nx_tilde)
		for i = 1 : N
			beq_gurobi_updated[(i-1)*nx_tilde+1:i*nx_tilde] = g_tilde_updated[:,i]
		end
		beq_gurobi_updated[1:nx_tilde] = beq_gurobi_updated[1:nx_tilde] + A_tilde_updated[:,:,1]*x_tilde_0 	# PARAMETER: depends on x0

		# FUTURE: implement references

		# println("H_gurobi:  $(H_gurobi)")
		# println("f_gurobi_init:  $(f_gurobi_init)")

	    GurobiEnv = Gurobi.Env()
		setparam!(GurobiEnv, "Presolve", -1)	# -1: automatic; example has 0; no big influence on solution time
#		setparam!(GurobiEnv, "LogToConsole", 0)	# # set presolve to 0 what does it mean?
		# setparam!(GurobiEnv, "TimeLimit",0.025)		# for 20Hz = 50ms

		# note that: (1/2) * z' * (2*H) * z + f' * z
	    GurobiModel = gurobi_model(GurobiEnv;
	    			name = "qp_01",
	    			H = 2*H_gurobi,
	    			f = f_gurobi_updated,		# updated b/c of reference updates
	    			Aeq = Aeq_gurobi_updated, 	# updated b/c A,B matrices new
	    			beq = beq_gurobi_updated,	# updated b/c g matrices and x0 new	
	    			lb = lb_gurobi,
	    			ub = ub_gurobi		)
	    optimize(GurobiModel)
	 	solvTimeGurobi = toq()

		println("Solving...")

		optimizer_gurobi = get_solution(GurobiModel)


		println("Solved...")

		status = get_status(GurobiModel)
		println(status)

		



  #       println("---- ey0, epsi0 in Gurobi:  $(x0') ---- ")
  #       println("u0 in Gurobi: $(u_0)")
		# println("s_pred in Gurobi (incl s0): $(s_pred)")
		# println("v_pred in Gurobi (incl v0): $(v_pred)")
		# println("k_coeffs in Gurobi: $(k_coeffs)")

		# structure of z = [ (da,ddf,X,Y,psi,vx,vy,wz,a,df) ; ....]
		da_pred_gurobi 	= optimizer_gurobi[1:n_uxu:end]
		ddf_pred_gurobi = optimizer_gurobi[2:n_uxu:end]
		X_pred_gurobi 	= optimizer_gurobi[3:n_uxu:end]
		Y_pred_gurobi 	= optimizer_gurobi[4:n_uxu:end]
		psi_pred_gurobi	= optimizer_gurobi[5:n_uxu:end]
		vx_pred_gurobi 	= optimizer_gurobi[6:n_uxu:end]
		vy_pred_gurobi 	= optimizer_gurobi[7:n_uxu:end]
		wz_pred_gurobi 	= optimizer_gurobi[8:n_uxu:end]
		a_pred_gurobi 	= optimizer_gurobi[9:n_uxu:end]
		df_pred_gurobi 	= optimizer_gurobi[10:n_uxu:end]

		state_pred_gurobi = hcat(X_pred_gurobi, Y_pred_gurobi, psi_pred_gurobi, vx_pred_gurobi, vy_pred_gurobi, wz_pred_gurobi)
		input_pred_gurobi = hcat(a_pred_gurobi, df_pred_gurobi)
		# println("ddf_pred_gurobi: $(ddf_pred_gurobi)")
		# println("df_pred_gurobi: $(df_pred_gurobi)")
		# println("ey_pred_gurobi (incl ey0): $(ey_pred_gurobi)")
		# println("epsi_pred_gurobi (incl v0): $(epsi_pred_gurobi)")

		# get first input
		a_opt = a_pred_gurobi[1]
		df_opt = df_pred_gurobi[1]

		# return whatever you want/need
	   	return a_opt, df_opt, status, solvTimeGurobi, state_pred_gurobi, input_pred_gurobi

	end  	# end of solve_gurobi()
end
