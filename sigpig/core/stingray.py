"""
Functions to interface with the Stingray ray tracing application. Also
contains functions to interface with NonLinLoc as part of the Stingray
workflow.

Stingray docs: https://pages.uoregon.edu/drt/Stingray/pages/26-48.html
NonLinLoc docs:
"""
from obspy import read, UTCDateTime, Inventory
import pandas as pd
from scipy.io import savemat
import numpy as np
import datetime
import matplotlib.pyplot as plt
from data import rattlesnake_Ridge_Station_Locations
from lidar import grids_from_raster, elevations_from_raster
import utm
import geopy.distance
import matlab.engine


# function to generate the necessary files for Stingray local earthquake tomography
def stingray_setup(project_name: str, date: UTCDateTime):
    """ Generates the necessary files to run Stingray for a specified project.
                       Stingray structures specified here:
              https://pages.uoregon.edu/drt/Stingray/pages/89.html
        A side effect of this function is that it also writes a
        station_locations.dat file containing station location information
        that is used by the stingray_ttg_to_nonlinloc function and
        stingray_ttg_2_nonlinloc.m.

    Example:
        # to get Rattlesnake Ridge station locations a date must be specified
        date = UTCDateTime("2018-03-16T00:04:00.0Z")
        project_name = "Rattlesnake Ridge"
        stingray_setup(project_name, date)
    """

    if project_name == "Rattlesnake Ridge":

        UTM_COOR = True

        srControl = False
        srGeometry = False
        srStation = True
        srEvent = False
        srModel = False
        srElevation = False

        if srControl:
            # generate srControl file
            # https://pages.uoregon.edu/drt/Stingray/pages/17-51.html
            condict = {}
            condict['tf_latlon'] = 1 if not UTM_COOR else 0
            condict['tf_anisotropy'] = 0 # include anisotropy in ray tracing forward problem?
            condict['tf_line_integrate'] = 0
            condict['arcfile'] = 'arc7.mat'
            condict['tf_waterpath'] = 0
            condict['tf_carve'] = 1
            condict['carve'] = {}
            condict['carve']['buffer'] = 0.2 # was 2
            condict['carve']['zvalue'] = [[]] # was [2]
            savemat("/Users/human/git/sigpig/sigpig/stingray/srInput"
                    "/srControl_TN.mat",
                {'srControl': condict})

        if srGeometry:
            # generate srGeometry file
            geodict = {}

            if UTM_COOR:
            # define lower left corner in utm
                geodict['tf_latlon'] = 0
                geodict['easting'] = 694.15
                geodict['northing'] = 5155.40
            else:
                geodict['tf_latlon'] = 1
                geodict['longitude'] = -120.46653901782679
                geodict['latitude'] = 46.52635322514362
                geodict['tf_flat'] = 0

            geodict['rotation'] = 0.0

            savemat(
                "/Users/human/git/sigpig/sigpig/stingray/srInput/srGeometry_TN"
                ".mat",
                {'srGeometry': geodict})

        if srStation:
            # -------------------------------------------------------------------
            # generates srStation mat file containing station locations
            if UTM_COOR:
                dfdict = {'name': [], 'northing': [], 'easting': [],
                          'elevation': []}
            else:
                dfdict = {'name': [], 'latitude': [], 'longitude': [],
                          'elevation': []}

            # get station locations on specified date from native GPS elevation
            station_locations = rattlesnake_Ridge_Station_Locations(date)

            # get station elevations from DEM rather than using native GPS
            # elevations
            raster_file = '/Users/human/Dropbox/Programs/lidar/yakima_basin_2018_dtm_43.tif'
            for station in station_locations.keys():
                elevation = elevations_from_raster(raster_file,
                                               [station_locations[station][1]],
                                               [station_locations[station][0]])
                station_locations[station][2] = elevation[0]

            # assemble dict in proper format for Stingray
            for station in station_locations.keys():
                if UTM_COOR:
                    easting, northing, _, _ = utm.from_latlon(station_locations[
                                                              station][0],
                                                              station_locations[
                                                              station][1])

                    # append eastings, northings, and elevations in km
                    dfdict['northing'].append(northing / 1000)
                    dfdict['easting'].append(easting / 1000)
                else:
                    dfdict['latitude'].append(station_locations[station][0])
                    dfdict['longitude'].append(station_locations[station][1])

                dfdict['elevation'].append(station_locations[station][2] /1000)

                # convert UGAP station names
                if station == "UGAP3":
                    station = "103"
                elif station == "UGAP5":
                    station = "105"
                elif station == "UGAP6":
                    station = "106"
                dfdict['name'].append([station])

            # write station locations to file
            with open('station_locations.dat', 'a') as file:
                for index in range(len(dfdict['name'])):
                    # write station, easting, northing, elevation
                    file.write(f'{dfdict["name"][index][0]} '
                               f'{dfdict["easting"][index]} '
                               f'{dfdict["northing"][index]} '
                               f'{dfdict["elevation"][index]}\n')

            dfdict['name'] = np.asarray(dfdict['name'], dtype=object)

            if UTM_COOR:
                dfdict['northing'] = np.asarray(dfdict['northing'], dtype=float)
                dfdict['easting'] = np.asarray(dfdict['easting'], dtype=float)
            else:
                dfdict['latitude'] = np.asarray(dfdict['latitude'], dtype=float)
                dfdict['longitude'] = np.asarray(dfdict['longitude'], dtype=float)

            dfdict['elevation'] = np.asarray(dfdict['elevation'], dtype=float)

            # save dict as mat file
            savemat(
                "/Users/human/git/sigpig/sigpig/stingray/srInput/srStation_TN"
                ".mat",
                {'srStation': dfdict})

        if srEvent:

            # -------------------------------------------------------------------
            # this makes srEvent file
            # TODO: edit this for RR - earthquake locations from NLL?
            # FIXME:
            # FIXME:
            # FIXME:
            # FIXME:
            # FIXME:
            # FIXME:
            # FIXME:
            # FIXME:
            # FIXME:
            # FIXME:
            # FIXME:
            # FIXME:
            # FIXME:
            # FIXME:
            # FIXME:
            # FIXME:

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
                "/Users/human/git/sigpig/sigpig/stingray/srInput/srEvent_TN"
                ".mat",
                {'srEvent': eqdict})

        if srModel:
            # generates srModel structure containing the slowness model

            # first specify grid information for srModel header in kilometers
            dx = dy = dz = 0.002 # 2 m model node spacing in all directions
            xoffset = 0
            yoffset = 0
            maxdep = 0.150 # 150 meters
            xdist = 0.300
            ydist = 0.500
            nx = int(xdist // dx + 1)
            ny = int(ydist // dy + 1)
            nz = int(maxdep // dz + 1)

            # load the velocity model
            # velmod = pd.read_csv("/Users/human/git/sigpig/sigpig/stingray/m-files/vels.txt", delim_whitespace=True)

            # # original velocity model from Doug
            # velocity_model = [[0.00, 0.60],
            #                   [0.05, 0.65],
            #                   [0.10, 0.70],
            #                   [0.15, 0.75]]

            # half-speed velocity model
            velocity_model = [[0.00, 0.30],
                              [0.05, 0.35],
                              [0.10, 0.40],
                              [0.15, 0.45]]

            velocity_model = pd.DataFrame(velocity_model, columns=['Top', 'Pvel'])

            # plot an interpolated velocity model
            plt.figure()
            plt.plot(velocity_model['Top'].values, velocity_model['Pvel'].values,
                     label='P model values')
            # plt.plot(velmod['Top'].values, velmod['Svel'].values,
            #          label='S model values')
            pz = np.polyfit(velocity_model['Top'].values, velocity_model['Pvel'].values, 1)
            # sz = np.polyfit(velocity_model['Top'].values, velocity_model['Svel'].values, 1)
            depth = np.linspace(0, maxdep, 100)
            plt.plot(depth, depth * pz[0] + pz[1], label='P interp values')
            # plt.plot(depth, depth * sz[0] + sz[1], label='S interp values')
            plt.legend()
            plt.show()

            # save the interpolated velocity model
            mod = np.concatenate((np.reshape(depth, (-1, 1)),
                                  np.reshape(depth * pz[0] + pz[1],
                                             (-1, 1))), axis=1)
                    # , np.reshape(depth * sz[0] + sz[1], (-1, 1))
            # np.savetxt('vels_linear.txt', mod, fmt='%6.5f', delimiter=' ')

            # make a 3D slowness model from the 1D velocity model
            Pmod = np.zeros((nx, ny, nz))
            # Smod = np.zeros((nx, ny, nz))
            for ii in range(nz):
                Pmod[:, :, ii] = 1 / ((ii * dz) * pz[0] + pz[1]) * np.ones(
                    (nx, ny))  # p slowness
                # Smod[:, :, ii] = 1 / ((ii * dz) * sz[0] + sz[1]) * np.ones(
                #     (nx, ny))  # s slowness

            # make dict of P wave slowness
            modeldict = {}
            modeldict['ghead'] = [xoffset, yoffset, nx, ny, nz, dx, dy, dz]
            modeldict['P'] = {}
            modeldict['P']['u'] = Pmod
            modeldict['S'] = {}
            modeldict['S']['u'] = [] # modeldict['S']['u'] = Smod
            savemat(
                "/Users/human/git/sigpig/sigpig/stingray/srInput/srModel_" + str(
                    int(1000 * dz)) + "m_TN.mat", {'srModel': modeldict})

        if srElevation:
            # generates stingray elevation .mat file from raster
            project_name = "Rattlesnake Ridge"
            elevation_map_from_arrays(project_name, UTM=UTM_COOR)

    else:
        pass

    return None


def elevation_map_from_arrays(project_name, UTM=False):
    """
    Generates an elevation map for Stingray as srElevation mat file
    containing topography.

    Example:
        project_name = "Rattlesnake Ridge"
        # get elevation map in UTM coordinates
        elevation_map_from_arrays(project_name, UTM=True)

    """
    if project_name == "Rattlesnake Ridge":
        # load arrays from raster file
        raster_file = '/Users/human/Dropbox/Programs/lidar/yakima_basin_2018_dtm_43.tif'

        if UTM:
            # doug's limits
            x_limits = [694.15, 694.45]
            y_limits = [5155.40, 5155.90]

            # get x and y distance in meters
            x_dist_m = (x_limits[1] - x_limits[0]) * 1000
            y_dist_m = (y_limits[1] - y_limits[0]) * 1000
            # x and y steps for loops
            num_x_steps = int(x_dist_m)  # 1 m resolution
            num_y_steps = int(y_dist_m)
            x_step = round((x_limits[1] - x_limits[0]) / num_x_steps, 3)
            y_step = round((y_limits[1] - y_limits[0]) / num_y_steps, 3)

        # in lat/lon
        else:
            # my limits
            # x_limits = [-120.480, -120.462]
            # y_limits = [46.519, 46.538]

            # doug's limits
            x_limits = [-120.4706347915009, -120.46074932200101]
            y_limits = [46.52239398104922, 46.530274799769188]

            # get x and y distance in feet
            x_dist_ft = geopy.distance.distance((y_limits[0], x_limits[0]),
                                             (y_limits[0], x_limits[1])).ft
            y_dist_ft = geopy.distance.distance((y_limits[0], x_limits[0]),
                                             (y_limits[1], x_limits[0])).ft
            # x and y steps for loops
            num_x_steps = int(x_dist_ft / 3) # dataset resolution is 3 ft
            num_y_steps = int(y_dist_ft / 3)
            x_step = (x_limits[1] - x_limits[0]) / num_x_steps
            y_step = (y_limits[1] - y_limits[0]) / num_y_steps

        # query raster on a grid
        longitude_grid, latitude_grid, elevation_grid = grids_from_raster(
                                raster_file, x_limits, y_limits, plot=True,
                                UTM=True)

        # define header
        elev_header = [x_limits[0], x_limits[1], y_limits[0], y_limits[1],
                       x_step, y_step, num_x_steps, num_y_steps]

        # build dict to make .mat file
        elev_dict = {}
        elev_dict['header'] = elev_header
        elev_dict['data'] = np.rot90(elevation_grid, k=3)

        savemat("/Users/human/git/sigpig/sigpig/stingray/srInput"
                "/srElevation_TN.mat",
                {'srElevation': elev_dict})

    return None

def stingray_ttg_to_nonlinloc(project_name):
    """
    Takes in Stingray-generated travel time grids (srRays files) and converts
    them to a travel time grid in the NonLinLoc format via a Matlab script.
    The NonLinLoc .hdr format is:
        Line 1:
            xNum, yNum, zNum: integers representing the number of grid nodes in
                              the x, y, and z directions
            xOrig, yOrig, xOrig: floats representing the x, y, z location of
                                 the grid origin in km relative to the
                                 geographic origin in Non-Global. If Global,
                                 longitude and latitude in degrees, and z in km
                                 of the location of the southwest corner of
                                 the grid.
            dx, dy, dz: floats representing the grid node spacing in
                        kilometers in the x, y, and z directions if Non-Global.
                        If Global, x and y in degrees, and z in kilometers.
            gridType: string
        Line 2:
            label: string representing the source or station label
            xSrce, ySrce: floats representing the x and y positions relative to
                          geographic origin in kilometers for source if
                          Non-Global. If Global, longitude and latitude in
                          degrees for source.
            zSrce: float representing the depth in kilometers for source

    Returns: None

    Example:
        project_name = "Rattlesnake Ridge"
        stingray_ttg_to_nonlinloc(project_name)
    """
    if project_name == "Rattlesnake Ridge":
        # open a matlab shell
        eng = matlab.engine.start_matlab()

        # run the stingray_ttg_2_nonlinloc.m file
        eng.stingray_ttg_2_nonlinloc(nargout=0)

        # stop the matlab engine
        eng.quit()

    return None

def nonlinloc_setup():
    """
    Function to generate files necessary to run NonLinLoc.

    """

    ...