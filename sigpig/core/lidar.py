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
from rasterio.fill import fillnodata
from affine import Affine
from pyproj import Proj, transform
from obspy.core.utcdatetime import UTCDateTime
from data import rattlesnake_Ridge_Station_Locations
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import tifffile as tiff
import scipy
import numpy.ma as ma


def visualize_Point_Cloud(filename):
    """
    Visualizes a .laz pointcloud file using pptk viewer.

    Example:
        filename = "/Users/human/Dropbox/Programs/lidar/RR_2019_10cm_NAD83_UTMz10.laz"
        filename = '/Users/human/Downloads/datasetsA2/yakima_basin_2018/laz/46120E4313.laz'
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
    v.set(point_size=0.0001, bg_color=[0.2,0,0,1], show_axis=0, show_grid=0)

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
    Reads a raster file and returns numpy arrays containing elevations,
    longitudes, and latitudes present in the raster file.

    Args:
        raster_file: string defining path to raster file

    Returns:
        elevations: np.ndarray of elevations in meters
        longitudes: np.ndarray of longitudes in WGS84
        latitudes: np.ndarray of longitudes in WGS84

    Example:
        raster_file = '/Users/human/Dropbox/Programs/lidar/GeoTiff/rr_dem_1m/hdr.adf'
        # or
        raster_file = '/Users/human/Dropbox/Programs/lidar/yakima_basin_2018_dtm_43.tif'

        elevations, longitudes, latitudes = arrays_from_raster(raster_file)
    """

    # read information from file
    with rasterio.open(raster_file) as r:
        # get upper-left pixel corner affine transform
        upper_left_pixel_transform = r.transform
        profile = r.profile
        raster_crs = Proj(r.crs)
        z = r.read(1)  # pixel values
        # interpolate to fill missing values
        z = fillnodata(z, mask=r.read_masks(1))

    # get rows and columns
    cols, rows = np.meshgrid(np.arange(z.shape[2]), np.arange(z.shape[1]))

    # get affine transform for pixel centres
    pixel_center_transform = upper_left_pixel_transform * Affine.translation(0.5, 0.5)
    # convert pixel row and col index to x and y coordinates at pixel center
    row_col_to_xy = lambda row, col: (col, row) * pixel_center_transform

    # get x and y coordinates in native crs (eastings and northings if utm)
    x, y = np.vectorize(row_col_to_xy, otypes=[float, float])(rows, cols)
    # constrain to 2D array for consistent array sizes
    z = z[0]

    # get longitudes and latitudes from native x and y coordinates
    lat_lon_crs = Proj(proj='latlong',datum='WGS84')
    longitudes, latitudes, elevations = transform(raster_crs, lat_lon_crs, x, y, z)

    # plot for testing
    a = longitudes[8000:13500, :5000]
    b = latitudes[8000:13500, :5000]
    c = elevations[8000:13500, :5000]
    # mask missing values
    a[c <= 0] = np.nan
    b[c <= 0] = np.nan
    c[c <= 0] = np.nan
    import matplotlib as mpl
    mpl.use('macosx')
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(a, b, c, cmap=cm.coolwarm, vmin=np.nanmin(c),
                           vmax=np.nanmax(c), linewidth=0, antialiased=False)
    ax.set_zlim(np.nanmin(c), np.nanmax(c)+1000)
    ax.xaxis.set_major_locator(LinearLocator(10))
    ax.yaxis.set_major_locator(LinearLocator(10))
    plt.xlim(max(x), min(x))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    newtif_file = r"dem_filled.tif"
    with rasterio.open(newtif_file, 'w', **profile) as dest:
        dest.write_band(1, z)

    return elevations, longitudes, latitudes


def elevations_from_arrays(elevations, longitudes, latitudes, query_points):
    """
    # TODO: double check that this function returns the expected result,
            something seems weird

    Finds the points closest to the corresponding points specified in
    query_points and queries the elevation at the closest point.

    Args:
        elevations: np.ndarray of elevations in meters
        longitudes: np.ndarray of longitudes in WGS84
        latitudes: np.ndarray of longitudes in WGS84
        query_points: list of lists comprising longitude latitude pairs

    Returns:
        list floats specifying queried elevations

    Example:
        # first get numpy arrays from a raster file
        raster_file = '/Users/human/Dropbox/Programs/lidar/yakima_basin_2018_dtm_43.tif'
        elevations, longitudes, latitudes = arrays_from_raster(raster_file)

        # build list of point to query data at
        query_points = [[-120.50126075151108, 46.624961220076976]]
        # then query the data at the list of points
        raster_elevations = elevations_from_arrays(elevations, longitudes, latitudes, query_points)
    """
    # helper function to build a cKD tree
    def kdtree(transformed_array, query_points):
        tree = scipy.spatial.cKDTree(transformed_array)
        _, indices = tree.query(query_points)

        return indices

    # transform data for kdtree
    transformed_array = np.dstack([latitudes.ravel(), longitudes.ravel()])[0]
    # get the closest points
    result = kdtree(transformed_array, query_points)

    return result


def elevations_from_raster(raster_file, coordinates):
    """
    Reads the specified raster file and queries it at the specified
    coordinates. The raster values at the queried points are returned in a
    list.

    NOTE: a known limitation of this function is that if a queried value is
          not "close enough" (it's arbitrary) to an existing value in the
          raster then this function returns the default "missing value"
          value, i.e. probably not the expected output. Be careful or fix it.

    Args:
        raster_file: string defining path to raster file
        coordinates: list of tuples of latitude, longitude pairs

    Returns:
        None

    Example:
        raster_file = '/Users/human/Dropbox/Programs/lidar/yakima_basin_2018_dtm_43.tif'


    Old example:
        # raster_file = '/Users/human/Dropbox/Programs/lidar/GeoTiff/rr_dem_1m/hdr.adf'
        date = UTCDateTime("2018-03-16T00:04:00.0Z")

        # get station coordinates in utm reference frame
        utm_station_dict = rattlesnake_Ridge_Station_Locations(date, format='utm')
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

        # write WGS84 station locations and elevations to file and plot
        station_locations = rattlesnake_Ridge_Station_Locations()
        stations = []
        ticks = []
        fig, ax = plt.subplots(figsize=(12, 7))
        with open("dem_station_elevations.csv", "w") as file:
            for index, station in enumerate(station_locations.keys()):
                latitude = station_locations[station][0]
                longitude = station_locations[station][1]
                elevation = elevations[index]
                line = f"{station},{latitude},{longitude},{elevation}\n"
                file.write(line)
                if index not in bad_value_indices[0]:
                    plt.scatter(index, differences[index], c='k')
                else:
                    plt.scatter(index, 0, c='r')
                stations.append(station)
                ticks.append(index)
        plt.xticks(ticks, stations, rotation=90)
        # get y ticks at interval of 1
        loc = plticker.MultipleLocator(base=1.0)
        ax.yaxis.set_major_locator(loc)
        plt.grid(True, which='both')
        plt.title('Differenced station elevations (meters)')
        plt.ylabel('Elevation difference (m)')
        plt.show()

    """
    # a place to store elevations
    elevations = []

    # read raster from file
    with rasterio.open(raster_file) as r:

        # get elevation at each coordinate
        for elevation in r.sample(coordinates):
            elevations.append(elevation[0])

    return elevations

def plot_tiff(tiff_file):
    """Plots a tiff file using imshow

    # FIXME: this function is broken

    Example:
        tiff_file = "/Users/human/Dropbox/Programs/lidar/yakima_basin_2018_dtm_43.tif"
        plot_tiff(tiff_file)

    """

    tfile = tiff.imread(tiff_file)
    tfile.shape
    tiff.imshow(tfile)
    plt.show()

    # Which band are you interested.
    # 1 if there is only one band
    band_of_interest = 1

    # Row and Columns of the raster you want to know
    # the value
    row_of_interest = 30
    column_of_interest = 50

    # open the raster and close it automatically
    # See https://stackoverflow.com/questions/1369526
    with rasterio.open(tiff_file) as dataset:
        band = dataset.read(band_of_interest)
        value_of_interest = band[row_of_interest, column_of_interest]

# TODO: extract values from yakima_basin_2018_dtm_43.tif

# TODO: add content from lidar/ortho_station_map here or to figures.py