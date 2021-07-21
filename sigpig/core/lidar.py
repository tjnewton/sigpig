"""
Functions to process lidar data.
"""

import laspy as lp
import numpy as np
# FIXME: pptk doesn't pip install
import pptk

def visualize_Point_Cloud(filename):
    """
    Visualizes a .laz pointcloud file using pptk viewer.

    Example:
        filename = "RR_2019_10cm_NAD83_UTMz10.laz"
        v = visualize_Point_Cloud(filename)

        # close point cloud viewer
        v.close()
    """
    # open laz file
    point_cloud=lp.read(filename)

    # store points and colors in np arrays
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
    # colors = np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose()

    # launch pptk viewer
    v = pptk.viewer(points)
    v.set(point_size=0.0005,bg_color=[0.2,0,0,1],show_axis=0,show_grid=0)

    return v

# TODO: add content from lidar/ortho_station_map here or to figures.py