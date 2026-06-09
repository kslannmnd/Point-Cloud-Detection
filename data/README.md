# Data directory

`r3d/` contains the Record3D/LiDAR recordings used for inference tests.

Each `.r3d` file is a sequence of frames. The client sends the `.r3d` file plus `frame_index` to the server, and the server reconstructs the selected depth frame into a point cloud before inference.
