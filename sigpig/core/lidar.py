"""
Functions to process lidar data.
"""

import laspy as lp
import numpy as np
from osgeo import gdal
from osgeo.gdalconst import *
import struct
import pptk
import rasterio
from affine import Affine
from pyproj import Proj, transform
from data import rattlesnake_Ridge_Station_Locations

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

def ingest_DEM(raster_file, output_file):
    """
    Reads an .adf raster file and outputs a .tiff file.

    # FIXME: this function is broken

    Example:
        raster_file = '/Users/human/Dropbox/Programs/lidar/GeoTiff/rr_dem_1m/hdr.adf'
        output_file = 'classified.tiff'
        ingest_DEM(raster_file, output_file)

    """

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
        value = round(value, 8)
        out_str = out_str + str(struct.pack('d', value))
    # Once classified, write the output raster
    # In the example, it's not possible to use the same output format than the input file, because GDAL is not able to write this file format. Geotiff will be used instead
    gtiff = gdal.GetDriverByName('GTiff')
    output_dataset = gtiff.Create(output_file, xsize, ysize, 4)
    output_dataset.SetProjection(projectionfrom)
    output_dataset.SetGeoTransform(geotransform)

    output_dataset.GetRasterBand(1).WriteRaster(0, 0, xsize, ysize, out_str)
    output_dataset = None

    return output_dataset


def arrays_from_raster(raster_file):
    """
    Reads a raster file and returns numpy ndarrays containing elevations (
    meters), UTM easting and northing in NAD83 UTM zone 10N (meters),
    and longitudes and latitudes.

    Args:
        raster_file: string defining path to raster file

    Returns:
        elevations: np.ndarray of elevations in meters
        eastings: np.ndarray of eastings in UTM meters
        northings: np.ndarray of northings in UTM meters
        longitudes: np.ndarray of longitudes in WGS84
        latitudes: np.ndarray of longitudes in WGS84

    Example:
        raster_file = '/Users/human/Dropbox/Programs/lidar/GeoTiff/rr_dem_1m/hdr.adf'
        elevations, eastings, northings, longitudes, latitudes = arrays_from_raster(raster_file)
    """

    # read raster from file
    with rasterio.open(raster_file) as r:
        # get upper-left pixel corner affine transform
        upper_left_pixel_transform = r.transform
        raster_crs = Proj(r.crs)
        elevations = r.read()  # pixel values

    # get rows and columns
    cols, rows = np.meshgrid(np.arange(elevations.shape[2]), np.arange(
                             elevations.shape[1]))

    # get affine transform for pixel centres
    pixel_center_transform = upper_left_pixel_transform * Affine.translation(
                                                                      0.5, 0.5)
    # convert pixel row and col index to easting and northing at pixel center
    row_col_to_easting_northing = lambda row, col: (col, row) * \
                                                         pixel_center_transform

    # get eastings and northings
    eastings, northings = np.vectorize(row_col_to_easting_northing, otypes=[
                                       float, float])(rows, cols)

    # get longitudes and latitudes from eastings and northings
    lat_lon_crs = Proj(proj='latlong',datum='WGS84')
    longitudes, latitudes = transform(raster_crs, lat_lon_crs, eastings,
                                      northings)

    # for consistent array sizes
    elevations = elevations[0]

    return elevations, eastings, northings, longitudes, latitudes


def elevations_from_raster(raster_file, utm_coordinates):
    """
    Reads the specified raster file and queries it at the specified
    coordinates. The raster values at the queried points are returned in a
    list.

    Args:
        raster_file: string defining path to raster file

    Returns:
        None

    Example:
        raster_file = '/Users/human/Dropbox/Programs/lidar/GeoTiff/rr_dem_1m/hdr.adf'

        # get station coordinates in utm reference frame
        utm_station_dict = rattlesnake_Ridge_Station_Locations("utm")
        # transform station dict into list of tuples of x,y coordinates
        utm_coordinates = []
        old_elevations = []
        for key in utm_station_dict.keys():
            # extract easting and northing
            easting, northing, old_elevation = utm_station_dict[key]
            utm_coordinates.append((easting, northing))
            old_elevations.append(old_elevation)

        # query raster at specified coordinates
        elevations = elevations_from_raster(raster_file, utm_coordinates)

        # calculate difference between old elevations and new elevations
        differences = np.asarray(elevations) - np.asarray(old_elevations)
        # find anomalous values and ignore them
        bad_value_indices = np.where(abs(differences) > 1000)
        if len(bad_value_indices) > 0:
            for index in bad_value_indices[0]:
                elevations[index] = old_elevations[index]

        # write WGS84 station locations and elevations to file
        station_locations = rattlesnake_Ridge_Station_Locations()
        with open("dem_station_elevations.csv", "w") as file:
            for index, station in enumerate(station_locations.keys()):
                latitude = station_locations[station][0]
                longitude = station_locations[station][1]
                elevation = elevations[index]
                line = f"{station},{latitude},{longitude},{elevation}\n"
                file.write(line)


    """
    # a place to store elevations
    elevations = []

    # read raster from file
    with rasterio.open(raster_file) as r:

        # get elevation at each coordinate
        for elevation in r.sample(utm_coordinates):
            elevations.append(elevation[0])

    return elevations

# TODO: add content from lidar/ortho_station_map here or to figures.py