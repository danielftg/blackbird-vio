## state-estimator
 - ego-state ($\mathcal{X}(t)$) - I should change from pose estimate in the ekf to relative pose estimate. Initialized at wherever, 
		 - body frame, live, filter, own thread, IPC, builds on and modifies [[gronhaug-state-est]] 
		   - thread: stream latest ego-state (as close to current real time)  
		 - EKF: 
			 Processes directly the measurements: RPM, IMU, feature points. Measurement model and noise comes from noise or is known. 
			 Process noise comes from env/control processes. 
			 See [[synth/modules/overview#Ego-state estimation|s-mskf]] for machinery (No points in state, bias,...)
			 See [[lynen-mod-multi-sens|Buffered state waits for measurements, live current state]] and [[forster-pre-integration|Pre-integration of high frequency measurements]] for the trailing state. 
		 - Need strong process model see [[svacha-quadcopter-state]] and [[wuest-online-param-est]]
		- [[geneve-openvins]] is conceptually similar similar to what we do. 

	 - environment state ($\mathcal{E}(t)$)
			 - Needs to support target tracking, obstacle avoidance, path planning and navigation for control. 
			- ought to support ego-state estimate through measurements: Pixel tracking, loocalization for pose and process covariance: contact/transitions for disutrbance
			  - map representation need to be 4D 
		 - Focus point logic. 
		 - Fixed time 3D reconstruction (Dense stereo) 
		 - Temporal 3D reconstruction (accumulation in time, temporal tracking) 
		 - Object segmentation and ID (coarse to fine segmentation) 
	 -The three preceding need to be steered by the focus point. 
		 - Localication and Navigation (Where am i now, and how do i get to this other place i've been) - Loop-closure
			
controller
	- low level stabilization: Betaflight FC board
	- st. tactical
	- lt.  strategic
	- FSM:        - Pursue
				- Survival > pursuit
			- Search 
				- Active perception, re-ID, belief of location. 
			- Engage
				- Constraints reduced massively. Survival < pursuit
			- Return 
				- Follow the map back with loop closure. 
			- Take-off / Landing 
				- Increase disturbance process covariance 
				- Exponential decay

	Planner
		Trajectories (task) and constraints (obstacles, cost)
	Resource control
		Focus point/radius,..
		CPU, MEM.. allocation? cgroupv2
	Motor control
		Trajectory -> motor commands
