To run Gazebo:
> roslaunch exercises gazebo_chase.launch

To run robot chase:
> roslaunch exercises robot_chase.launch mode_baddies:=random mode_police:=closest
or
> roslaunch exercises robot_chase.launch mode_baddies:=potential_field mode_police:=potential_field

To run RViz:
> roslaunch exercises rviz.launch

Folders:

worlds - simple.world, loads scene and physics
models - What should be models is actually linked to models in part 1. Modify worlds in part1/models. model.config loads model.sdf where walls and obstacles are specified
launch - gazebo_chase.launch:
		- includes file empty_world.launch with arg simple.world (/opt/ros/kinetic/gazebo_ros/launch)
		- sets robot_description
		- Includess robots.launch: Create a node for each robot in its own namespace of package gazebo_ros with model turtlebot3
	- robot_chase:
		- Spawns node of type robot chase with modes parameters as above
