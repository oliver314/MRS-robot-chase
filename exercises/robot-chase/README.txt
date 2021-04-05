To run Gazebo:
> roslaunch exercises gazebo_chase.launch nr_baddies:=3 nr_police:=3 world:=simple

To run RViz:
> roslaunch exercises rviz_chase.launch

To run robot chase:
> roslaunch exercises robot_chase.launch mode_baddies:=BADDIES_MODE mode_police:=POLICE_MODE nr_baddies:=NR_BADDIES nr_police:=NR_POLICE

e.g.
roslaunch exercises robot_chase.launch mode_baddies:=potential_field mode_police:=potential_field nr_baddies:=3 nr_police:=3

Modes for Baddies:
'random', 'potential_field', 'est_test

Modes for Police:
'closest', 'potential_field', 'est_test'

Folders:roslaunch exercises robot_chase.launch mode_baddies:=BADDIES_MODE mode_police:=BADDIES_MODE

worlds - simple.world, loads scene and physics. Contains folders with models. model.config loads model.sdf where walls and obstacles are specified
launch - gazebo_chase.launch:
	    - includes file empty_world.launch with arg simple.world (/opt/ros/kinetic/gazebo_ros/launch)
		- sets robot_description
		- Includess robots.launch: Create a node for each robot in its own namespace of package gazebo_ros with model turtlebot3
    - robot_chase:
		- Spawns node of type robot chase with modes parameters as above
        - Deals with setting all the transforms and remapping of topics such that it can be viewed easilly in RViz

    - rviz_chase:
        - Opens the rviz window with the configuration saved in model.rviz (/catkin_ws/src/turtlebot3/turtlebot3_description/rviz)
