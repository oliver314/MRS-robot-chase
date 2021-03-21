To run Gazebo:
> roslaunch exercises gazebo_chase.launch

To run obstacle avoidance:
> roslaunch exercises robot_avoidance.launch mode:=braitenberg

To run RViz:
> roslaunch exercises rviz.launch

Folders:

worlds - simple.world, loads scene and physics
models - model.config loads model.sdf where walls and obstacles are specified
launch - gazebo_chase.launch:
		- includes file empty_world.launch with arg simple.world (/opt/ros/kinetic/gazebo_ros/launch)
		- sets robot_description
		- Includess robots.launch: Create a node for each robot in its own namespace of package gazebo_ros with model turtlebot3
	- robot_avoidance:
		- Spawns node of type robot avoidance with parameter mode (braitenberger, etc)
