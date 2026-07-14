


## Cheap primitives
Histogram methods / Binning
Image filters: Sobel, etc.
Coordinate transforms: RGB -> LAB


## Ego-state estimation
[[sun-s-msckf|Kalman filter variant using IMU and stereo image measurements]]
	Contains alternative ways to represent points in the filter.
	Contains different KF variants and choices. 
[[lynen-mod-multi-sens|Buffered state waits for measurements, live current state]]
[[forster-pre-integration|Pre-integration of high frequency measurements]]

[[svacha-quadcopter-state|Strong process model to support aggressive, high speed flight]]
[[wuest-online-param-est|Online offsets and parameter est]]


## Environment estimation

### Mapping 
[[campos-orb-slam3|static world mapping, loop and map merging (loop correction, place recognition)]]

### Matching
[[tarrio-edge-vo|edge tracking and matching may be useful for stereo matching]]

[[engel-dso|photometric error is an alternative matching strategy requiring photometric calibration.]]
	[[forster-svo|More photometric ]]

### Mesh model 
[[combe-dyn-fusion|Provides a way to produce a surface model of a moving object in the scene for a moving camera]]


## Control
[[saviolo-nova|Pursuit control and some perception hints]]
		How to choose dynamically feasible, optimal, obstacle avoidant trajectories. 

### Search 
[[saviolo-hunt|Search mode and logic]]
	"loitering mode"
