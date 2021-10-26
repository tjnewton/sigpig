"""
Functions to interface with the Stingray ray tracing application.
Stingray docs: https://pages.uoregon.edu/drt/Stingray/pages/26-48.html

"""
from obspy import read, UTCDateTime, Inventory
import pandas as pd
from scipy.io import savemat
import numpy as np
import datetime
import matplotlib.pyplot as plt
from data import rattlesnake_Ridge_Station_Locations
from lidar import arrays_from_raster


# function to generate the necessary files for Stingray local earthquake tomography
def stingray_setup(project_name: str, date: UTCDateTime):
    """ Generates the necessary files to run Stingray for a specified project

    Example:
        # to get Rattlesnake Ridge station locations a date must be specified
        date = UTCDateTime("2018-03-16T00:04:00.0Z")
        project_name = "Rattlesnake Ridge"
        stingray_setup(project_name, date)
    """

    if project_name == "Rattlesnake Ridge":

        srControl = False
        srGeometry = False
        srStation = False
        srEvent = False
        srModel = True
        srElevation = True

        if srControl:
            # generate srControl file
            # https://pages.uoregon.edu/drt/Stingray/pages/17-51.html
            condict = {}
            condict['tf_latlon'] = 1 # 1 = geodetic coordinates, 0 = UTM coordinates
            condict['tf_anisotropy'] = 0 # include anisotropy in ray tracing forward problem?
            condict['tf_line_integrate'] = 0
            condict['arcfile'] = 'arc7.mat'
            condict['tf_waterpath'] = 0
            condict['tf_carve'] = 1
            condict['carve'] = {}
            condict['carve']['buffer'] = 2
            condict['carve']['zvalue'] = [[2]]
            savemat("/Users/human/Dropbox/Programs/stingray/projects/rattlesnake_ridge/srInput/srControl.mat",
                {'srControl': condict})

        if srGeometry:
            # generate srGeometry file
            geodict = {}
            geodict['longitude'] = -120.4668
            geodict['latitude'] = 46.5253
            geodict['rotation'] = 0.0
            geodict['tf_latlon'] = True
            geodict['tf_flat'] = 0.0
            savemat(
                "/Users/human/Dropbox/Programs/stingray/projects/rattlesnake_ridge/srInput/srGeometry.mat",
                {'srGeometry': geodict})

        if srStation:
            # -------------------------------------------------------------------
            # generates srStation mat file containing station locations
            dfdict = {'name': [], 'latitude': [], 'longitude': [],
                      'elevation': []}

            # get Rattlesnake Ridge station locations on specified date
            station_locations = rattlesnake_Ridge_Station_Locations(date)
            # assemble dict in proper format for Stingray
            for station in station_locations.keys():
                dfdict['name'].append([station])
                dfdict['latitude'].append([station_locations[station][0]])
                dfdict['longitude'].append([station_locations[station][1]])
                dfdict['elevation'].append([station_locations[station][2]])
            dfdict['name'] = np.asarray(dfdict['name'], dtype=object)
            dfdict['latitude'] = np.asarray(dfdict['latitude'], dtype=object)
            dfdict['longitude'] = np.asarray(dfdict['longitude'], dtype=object)
            dfdict['elevation'] = np.asarray(dfdict['elevation'], dtype=object)

            # save dict as mat file
            savemat(
                "/Users/human/Dropbox/Programs/stingray/projects/rattlesnake_ridge/srInput/srStation.mat",
                {'srStation': dfdict})

        if srEvent:

            # -------------------------------------------------------------------
            # this makes srEvent file
            # TODO: edit this for RR - earthquake locations from NLL
            eqdf = pd.read_csv(
                "/Users/human/Dropbox/Programs/stingray/projects/rattlesnake_ridge/earthquakes.csv")

            datetimes = pd.to_datetime(eqdf['DateTime'])
            origins = np.zeros((len(eqdf), 1))
            for ii in range(len(eqdf)):
                origins[ii] = \
                (datetimes - datetime.datetime.utcfromtimestamp(0)).values[
                    ii].item() / 1e9
            eqdict = {}
            eqdict['id'] = eqdf['EventID'].values.reshape(len(eqdf), 1)
            eqdict['type'] = 3 * np.ones((len(eqdf), 1))
            eqdict['latitude'] = eqdf['Latitude'].values.reshape(len(eqdf), 1)
            eqdict['longitude'] = eqdf['Longitude'].values.reshape(len(eqdf),
                                                                   1)
            eqdict['origintime'] = origins
            eqdict['depth'] = eqdf['Depth'].values.reshape(len(eqdf), 1)
            eqdict['datum'] = np.zeros((len(eqdf), 1))
            savemat(
                "/Users/human/Dropbox/Programs/stingray/projects/rattlesnake_ridge/srInput/srEvent.mat",
                {'srEvent': eqdict})

        # -------------------------------------------------------------------
        # generates srElevation mat file containing topography

        # -------------------------------------------------------------------

        if srModel:
            # TODO: edit this for RR
            #       - ask Doug for srModel.mat
            #       - what is appropriate model node spacing?
            #       - what is appropriate grid size?
            # this makes srModel file
            # set some options
            for d in [.25]:  # , .5, 1]:
                dx = dy = dz = d  # model node spacing, x-dir
                # model node spacing, y-dir
                # model node spacing, z-dir
                xoffset = -100.0
                yoffset = -100.0
                maxdep = 100.0
                xdist = 200.0
                ydist = 200.0
                nx = int(np.ceil(xdist) // dx)
                ny = int(np.ceil(ydist) // dy)
                nz = int(maxdep // dz + 1)

                # this makes the model file
                velmod = pd.read_csv("gil7.txt", delim_whitespace=True)
                plt.figure()
                plt.plot(velmod['Top'].values, velmod['Pvel'].values,
                         label='P model values')
                plt.plot(velmod['Top'].values, velmod['Svel'].values,
                         label='S model values')
                pz = np.polyfit(velmod['Top'].values, velmod['Pvel'].values, 1)
                sz = np.polyfit(velmod['Top'].values, velmod['Svel'].values, 1)
                depth = np.arange(100)
                plt.plot(depth, depth * pz[0] + pz[1], label='P interp values')
                plt.plot(depth, depth * sz[0] + sz[1], label='S interp values')
                plt.legend()

                mod = np.concatenate((np.reshape(depth, (-1, 1)),
                                      np.reshape(depth * pz[0] + pz[1],
                                                 (-1, 1)),
                                      np.reshape(depth * sz[0] + sz[1],
                                                 (-1, 1))), axis=1)
                np.savetxt('gil7_linear', mod, fmt='%6.5f', delimiter=' ')

                Pmod = np.zeros((nx, ny, nz))
                Smod = np.zeros((nx, ny, nz))
                for ii in range(nz):
                    Pmod[:, :, ii] = 1 / ((ii * dz) * pz[0] + pz[1]) * np.ones(
                        (nx, ny))  # p slowness
                    Smod[:, :, ii] = 1 / ((ii * dz) * sz[0] + sz[1]) * np.ones(
                        (nx, ny))  # s slowness

                # make dict
                modeldict = {}
                modeldict['ghead'] = [xoffset, yoffset, nx, ny, nz, dx, dy, dz]
                modeldict['P'] = {}
                modeldict['P']['u'] = Pmod
                modeldict['S'] = {}
                modeldict['S']['u'] = Smod
                savemat(
                    "/Users/human/Dropbox/Programs/stingray/projects/rattlesnake_ridge/srInput/srModel_" + str(
                        int(1000 * d)) + ".mat", {'srModel': modeldict})

    else:
        pass

    return None


def elevation_map_from_arrays(project_name, elevations, longitudes,
                              latitudes):
    """
    Generates an elevation map for Stingray from the specified numpy arrays.

    # TODO: build function and docstring

    Example:
        # first get numpy arrays from a raster file
        raster_file = '/Users/human/Dropbox/Programs/lidar/yakima_basin_2018_dtm_43.tif'
        elevations, longitudes, latitudes = arrays_from_raster(raster_file)

    """
    if project_name == "Rattlesnake Ridge":
        # load arrays from raster file
        raster_file = '/Users/human/Dropbox/Programs/lidar/yakima_basin_2018_dtm_43.tif'
        elevations, longitudes, latitudes = arrays_from_raster(raster_file)

        # define area of interest
        x_min =
        x_max =
        y_min =
        y_max =
        x_inc =
        y_inc =
        nx =
        ny =
        elev_header = [x_min, x_max, y_min, y_max, x_inc, y_inc, nx, ny]

        # trim to area of interest

        # interpolate values?

        # query values

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
                               vmax=np.nanmax(c), linewidth=0,
                               antialiased=False)
        ax.set_zlim(np.nanmin(c), np.nanmax(c) + 1000)
        ax.xaxis.set_major_locator(LinearLocator(10))
        ax.yaxis.set_major_locator(LinearLocator(10))
        plt.xlim(max(x), min(x))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    ...