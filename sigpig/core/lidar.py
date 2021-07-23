"""
Functions to process lidar data.
"""

import laspy as lp
import numpy as np
from osgeo import gdal
from osgeo.gdalconst import *
import struct
import pptk

def visualize_Point_Cloud(filename):
    """
    Visualizes a .laz pointcloud file using pptk viewer.

    Example:
        filename = "/Users/human/Dropbox/Programs/lidar/RR_2019_10cm_NAD83_UTMz10.laz"
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

def ingest_DEM(adm_filename):
    """

    """
    raster_file = '/Users/human/Dropbox/Programs/lidar/GeoTiff/rr_dem_1m/hdr.adf'
    output_file = 'classified.tiff'

    classification_values = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500,
                             4000]  # The interval values to classify
    classification_output_values = [10, 20, 30, 40, 50, 60, 70, 80,
                                    90]  # The value assigned to each interval

    # Opening the raster file
    dataset = gdal.Open(raster_file, GA_ReadOnly)
    band = dataset.GetRasterBand(1)
    # Reading the raster properties
    projectionfrom = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()
    xsize = band.XSize
    ysize = band.YSize
    datatype = band.DataType

    # Reading the raster values
    values = band.ReadRaster(0, 0, xsize, ysize, xsize, ysize, datatype)
    # Conversion between GDAL types and python pack types (Can't use complex integer or float!!)
    data_types = {'Byte': 'B', 'UInt16': 'H', 'Int16': 'h', 'UInt32': 'I',
                  'Int32': 'i', 'Float32': 'f', 'Float64': 'd'}
    values = struct.unpack(
        data_types[gdal.GetDataTypeName(band.DataType)] * xsize * ysize,
        values)

    # Now that the raster is into an array, let's classify it
    out_str = ''
    for value in values:
        index = 0
        for cl_value in classification_values:
            if value <= cl_value:
                out_str = out_str + struct.pack('B',
                                                classification_output_values[
                                                    index])
                break
            index = index + 1
    # Once classified, write the output raster
    # In the example, it's not possible to use the same output format than the input file, because GDAL is not able to write this file format. Geotiff will be used instead
    gtiff = gdal.GetDriverByName('GTiff')
    output_dataset = gtiff.Create(output_file, xsize, ysize, 4)
    output_dataset.SetProjection(projectionfrom)
    output_dataset.SetGeoTransform(geotransform)

    output_dataset.GetRasterBand(1).WriteRaster(0, 0, xsize, ysize, out_str)
    output_dataset = None

    return

# TODO: add content from lidar/ortho_station_map here or to figures.py