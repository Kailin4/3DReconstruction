# 3D Reconstruction Project

createPointCloud.py is the script that will be run to load the images. <br/>
It calls the functions that are responsible for processing the images. <br/>
To run your own dataset you will need to provide images and camera intrinsics.  <br/>
You will also need to modify the directory parameters and file extensions in the script.  <br/>
To test our pipeline you can run python createPointCloud.py. <br/>

helpPointCloud.py contains the helper functions for creating the point cloud. <br/><br/>
Connections.py contains the class to store information between all pair of views. <br/><br/>
Match.py contains the class to store the relationship between a single 2D feature with multiple views
and its associated 3D representations.

The images are in a submodule, run git submodule update --remote --init.

