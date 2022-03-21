"""
Picks first arrivals via unet and clusters arrivals to associate them to
events utilizing rules in a decision tree.
"""
# FIXME: address FIXME's and TODO's throughout code
# FIXME: verify model is only receiving P picks
# FIXME: test association on all 5s windows in 30s, and other subsets
# FIXME: change camel case to lowercase:::within function
# FIXME: there is a bug in here somewhere that generates duplicate phase
#        time picks in events with different hashs. This is probably due to
#        slightly different medians of duplicate pick times generating a unique
#        hash since the event_id is hashed from the median phase pick time.

# Run the autopicker?
RUN = False

 # # # # # # # # # #  N O T E S  # # # # # # # # # # #
# -------------------------------------------------- #
###### Counts for 10-min duplicate-removed runs ######
#            |   threshold   | picks |
#            | 0.23 from F1  |  9,056
#            | 0.30          |  6,821
#            | 0.40          |  4,797
#            | 0.50          |  3,364
#            | 0.60          |  2,270
#            | 0.70          |  1,309
#            | 0.80          |    560
#            | 0.90          |     83
#            | 0.95          |      3

# moveout across 16 stations (20 to 35)
# 0.04443 s -> 11 samples

# moveout across 20 stations (20 to 39)
# 0.1044 s -> 26 samples
# 0.183 s for different event

# required imports
import numpy as np
import obspy
from obspy import read, read_inventory, Stream
from obspy.core.utcdatetime import UTCDateTime
import pickle
import time
import unet
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator, num2date
from sklearn import decomposition
from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan
from collections import defaultdict
import base64
import hashlib
from datetime import datetime
import sys

# supress np error that is handled
np.seterr(divide='ignore', invalid='ignore')

# # # #  D E F I N E   P A R A M E T E R S  # # # #
# plot association results? only do this for small time periods, otherwise
# the figure will squished and useless.
plot_Association_Results = False # see warning above
load = False # load windows from a previous run?
detection_Parameters = (120, 60) # window size and offset

# Parameters for unet detection
threshold = 0.23 # was 0.40
sampling_Rate = 250 # samples per second
duplicate_Threshold = 4 # in samples. prev: 25,

# Transform time dimension for clustering?
scale_Time = True
scale_Factor = 4.0

# Perform principal component analysis for clustering?
PCA = False
PCA_Components = 3

# Choose type(s) of clustering to perform and associated parameters:
# ~~~~~~~~~ D B S C A N ~~~~~~~~~ #
_DBSCAN = True
# too high = merged events and skewed cluster medians for repicking
# too low = multiple clusters for a single event
DBSCAN_Distance = 0.73 # 0.59 - 0.80, 13 sec cluster splitting from .73-.74
DBSCAN_Min_Samples = 3
# ~~~~~~~~~ O P T I C S ~~~~~~~~~ #
_OPTICS = False
OPTICS_Min_Samples = 2 # 5
OPTICS_Xi = 0.06 # 0.05
OPTICS_Min_Cluster_Size = 4
OPTICS_Max_Eps = 0.60 # np.inf
# to compare with DBSCAN results
DBSCAN_Eps_1 = 0.50
DBSCAN_Eps_2 = 0.60
# ~~~~~~ S P E C T R A L ~~~~~~ #
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html
# ~~~ G A U S S I A N   M I X T U R E ~~~ #
# ~~~~~~ M E A N    S H I F T ~~~~~~ #
# ~~~~~~ K - P L A N E ~~~~~~ #
# FIXME: add other clustering algos

# Association parameters
temporal_Threshold = 0.14 # in seconds, based on maximum observed moveout time
spatial_Threshold = 3 # greater than N stations away = outside space threshold
min_Quakes_Per_Cluster = 5

# # #  H E L P E R     F U N C T I O N S  # # #
# function to return project stations active on a given date
def project_stations(project_name: str, start_time: UTCDateTime):
    """Returns a list of the stations active on the specified date for the
    specified project in the linear scarp reference frame from ~North to
    ~South.
    """
    if project_name == "Rattlesnake Ridge":
        # build filepath list based on dates
        date = datetime(2018, start_time.month, start_time.day)
        if date <= datetime(2018, 4, 8):
            stas = [1, 2, 3, 5, 4, 6, 7, 8, 13, 9, 10, 12, 15, 'UGAP3', 16, 17,
                    18, 20, 21, 22, 23, 25, 26, 27, 'UGAP5', 'UGAP6', 28, 30,
                    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
        elif date <= datetime(2018, 5, 7):
            stas = [1, 2, 3, 5, 4, 6, 7, 8, 13, 9, 10, 12, 14, 15, 'UGAP3', 16,
                    17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 'UGAP5', 'UGAP6',
                    28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
        elif date == datetime(2018, 5, 8):
            stas = [1, 3, 5, 4, 6, 7, 8, 13, 9, 10, 12, 14, 15, 'UGAP3', 16,
                    17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 'UGAP5', 'UGAP6',
                    28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
        elif date <= datetime(2018, 6, 6):
            stas = [1, 3, 5, 4, 6, 7, 8, 13, 9, 10, 12, 14, 15, 'UGAP3', 16,
                    17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 'UGAP5', 'UGAP6',
                    28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
        elif date == datetime(2018, 6, 7):
            stas = [1, 3, 5, 4, 6, 7, 8, 13, 9, 10, 14, 15, 'UGAP3', 16, 18,
                    20, 21, 22, 23, 24, 25, 26, 27, 'UGAP5', 'UGAP6', 28, 29,
                    30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42]
        else:
            stas = [1, 2, 3, 5, 4, 6, 7, 8, 13, 9, 10, 14, 15, 'UGAP3', 16, 18,
                    20, 21, 22, 23, 24, 25, 26, 27, 'UGAP5', 'UGAP6', 28, 29,
                    30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42]

    return stas


# function to return project filepaths
def project_Filepaths(project_Name: str, start_Time: UTCDateTime, end_Time: UTCDateTime):
    """Returns list of strings of filepaths for specified project, assuming
    files span a single day and start at T00.00.00.

    Parameters:
        project_Name: list
            list of strings that represent mseed filepaths

    Example:
        filepaths = project_Filepaths("Rattlesnake Ridge")
    """
    if project_Name == "Rattlesnake Ridge":
        # build filepath list based on dates
        stas = project_stations(project_Name, start_Time)

        # build station:channel dict from list of stations
        node = ['DP1'] #, 'DP2', 'DPZ']  # nodal seismometer channels
        ugap = ['EHN'] #, 'EHE', 'EHZ']
        stations_channels = {}
        for station in stas:
            if isinstance(station, int):
                stations_channels[str(station)] = node
            elif isinstance(station, str):
                stations_channels[station] = ugap
            else:
                print("You did something weird in the station list.")

        # stations_channels = {'1': node, '2': node, '3': node, '4': node,
        #                      '5': node, '6': node, '7': node, '8': node,
        #                      '9': node, '10': node, '12': node, '13': node,
        #                      '15': node, '16': node, '17': node, '18': node,
        #                      '20': node, '21': node, '22': node, '23': node,
        #                      '25': node, '26': node, '27': node, '28': node,
        #                      '30': node, '31': node, '32': node, '33': node,
        #                      '34': node, '35': node, '36': node, '37': node,
        #                      '38': node, '39': node, '40': node, '41': node,
        #                      '42': node, 'UGAP3': ugap, 'UGAP5': ugap,
        #                      'UGAP6': ugap}

        # assemble filepaths
        filepaths = []
        for station in stations_channels:
            for channel in stations_channels[station]:
                filepath = f"/Users/human/Desktop/RR_MSEED/5A.{station}.." \
                           f"{channel}.{start_Time.year}-" \
                           f"{start_Time.month:02}-{start_Time.day:02}T00.00.00.ms"
                # save filepath to list
                filepaths.append(filepath)

    return filepaths

def trace_arrival_prediction(trace, center_time, model):
    """ Takes in an Obspu trace and a time, then transforms the data to get
    the u-net arrival time prediction for a 120 sample window centered on
    the specified UTCDateTime, then returns the corresponding prediction
    array from the specified tensorflow model.

    """
    # for log calculation below
    epsilon = 1e-6

    # build picking windows from trace
    trace.trim(center_time - (59/250), center_time + (60/250))
    data = trace.data

    # first create a n x 2 array of zeros
    reshaped_picking_window = np.zeros((len(data), 2))

    # then loop through each row and fill it out
    for row_index in range(0, len(reshaped_picking_window)):
        # store the sign
        reshaped_picking_window[row_index][1] = np.sign(data[row_index])
        # store the amplitude
        reshaped_picking_window[row_index][0] = np.log(np.abs(data[row_index])\
                                       + epsilon) # epsilon avoids log(0) error

    # expand dimension of single picking window array so it is compatible
    # with the model
    reshaped_picking_window = np.expand_dims(reshaped_picking_window, axis=0)
    # get unet predictions from specified model
    pick_prediction = get_Unet_Picks(reshaped_picking_window,
                                      preloaded_model=model)

    return pick_prediction

    
# function to generate numpy array of log modulus transformed data for unet
# input (for first-arrival picking) for all stations. See
def find_Picking_Windows(filepaths: list, detection_Parameters:
                       tuple, time_Periods: list, filter=False, 
                       response=False, save_file=False) -> np.ndarray:
    """
    Requires files that are Obspy readable, containing data for a single
    station and channel. Different channels of data should be in separate files.
    Uses project_Name for preprocessing steps (use this as a template for
    new projects). A known limitation of this function is that it expects
    all time periods to occur on the same day.

    Parameters:
        filepaths: list of filepaths to mseed files
        detection_Parameters: tuple containing values for the window length 
                            in samples and the window offset in samples.
        time_Periods: list containing tuples of (start_Time, end_Time) for each
                      time period. Multiple time periods are represented as 
                      separate list entries, e.g [(start_Time_1, 
                      end_Time_1), (start_Time_2, end_Time_2)]. Where  
                      start_Time and end_Time are:
                        start_Time: UTCDateTime
                            defines the start time of the period of interest.
                        end_Time: UTCDateTime
                            defines the end time of the period of interest

    Example:
        detection_Parameters = (120, 60)                   
        start_Time = UTCDateTime("2018-03-13T01:33:00.0Z")
        end_Time =   UTCDateTime("2018-03-13T01:33:00.48Z")
        filepaths = project_Filepaths("Rattlesnake Ridge", start_Time, end_Time)
        time_Periods = [(start_Time, end_Time)]
        picking_Windows, picking_Windows_Untransformed, picking_Windows_Times = find_Picking_Windows(filepaths, detection_Parameters, time_Periods)
    
    """

    # set window parameters based on unpacked input
    window_Length, window_Offset = detection_Parameters

    # initialize empty list to store picking windows and their times
    picking_Windows = []
    picking_Windows_Untransformed = []
    picking_Windows_Times = []

    # loop through filepaths
    for filepath in filepaths:

        # define the station name and channel from filepath
        # my filepaths look like the following:
        # /Users/human/Desktop/RR_MSEED/5A.27..DP1.2018-03-13T00.00.00.ms
        # so I extract the station from the filename.
        # First, isolate the filename from the filepath (assuming filename
        # doesn't contain /)
        filename = filepath.split("/")[-1]
        # then grab the station name from the filename
        station = filename.split(".")[1]
        channel = filename.split(".")[3]
        # print(f" Processing station: {station}, channel: {channel}")
        # initialize empty list to store picking windows unique to this station and channel
        station_Picking_Windows = []
        station_Picking_Windows_Times = []

        # for log calculation below
        epsilon = 1e-6

        # loads data, merges traces if there is more than one, selects first trace
        trace = read(filepath).merge(method=1, fill_value=0)[0]

        # check if interpolation is needed
        if trace.stats.sampling_rate != sampling_Rate:
            starttime = time_Periods[0][0] - 60
            endtime = time_Periods[-1][1] + 60
            npts = int((endtime - starttime) * sampling_Rate)
            trace.interpolate(sampling_Rate, method="lanczos",
                                starttime=starttime,
                              endtime=endtime, npts=npts,a=30)

        # loop through time periods, ASSUMING they all occur on the same day
        for start_Time, end_Time in time_Periods:

            # copy trace and trim to specified time period
            period_Trace = trace.copy()
            period_Trace = period_Trace.trim(start_Time, end_Time)

            # correct for instrument response?
            if response:
                if period_Trace.stats.network =="UW":
                    invfile = \
                        '/Users/human/git/noisepicker-private/resps/RESP.' + period_Trace.stats.station \
                              + '.' + period_Trace.stats.network + '..' + period_Trace.stats.channel
                else:
                    invfile = '/Users/human/git/noisepicker-private/resps/RESP.' + period_Trace.stats.network \
                              + '.' + period_Trace.stats.station + '..' + period_Trace.stats.channel
                inv = read_inventory(path_or_file_object=invfile, format='RESP')
                period_Trace.remove_response(inventory=inv, output='VEL', plot=False)

            # bandpass filter and detrend data?
            if filter:
                period_Trace.detrend('linear')
                period_Trace.filter("bandpass", freqmin=20, freqmax=60,
                                    corners=4)
            
            times = period_Trace.times("utcdatetime")
            data = period_Trace.data

            # loop through each time step and grab windows of the specified length in samples
            for index in range(0, len(data) - window_Length, window_Offset):
                # print(f"Station: {station}  Channel: {channel}  : : :"
                #       f" {index} of {len(data)}")

                # extract start and end time for window
                window_Start = times[index]
                window_End = times[index + window_Length]

                # append picking window and metadata to lists
                station_Picking_Windows.append(data[index: (
                        index + window_Length)])  # append data within window
                station_Picking_Windows_Times.append((window_Start,
                                        window_End, station, channel))

            # append picking windows one at a time to major list
            # print(f"{start_Time}----{end_Time}      Found"
            #       f" {len(station_Picking_Windows)} picking windows.")

            for window_index in range(len(station_Picking_Windows)):
                # reshape array to (n, 2), where first dimension is
                # amplitude, and second dimension is sign

                # first create a n x 2 array of zeros
                reshaped_Picking_Window = np.zeros((len(
                    station_Picking_Windows[window_index]), 2))

                # store the original data
                picking_Windows_Untransformed.append(
                                        station_Picking_Windows[window_index])

                # then loop through each row and fill it out
                for row_Index in range(0, len(reshaped_Picking_Window)):

                    # store the sign
                    reshaped_Picking_Window[row_Index][1] = np.sign(
                        station_Picking_Windows[window_index][row_Index])

                    # store the amplitude
                    reshaped_Picking_Window[row_Index][0] = np.log(np.abs(
                        station_Picking_Windows[window_index][row_Index]) +
                        epsilon) # epsilon avoids log(0) error


                picking_Windows.append(reshaped_Picking_Window)
                picking_Windows_Times.append(
                    station_Picking_Windows_Times[window_index])

    # convert to np.ndarray
    picking_Windows = np.array(picking_Windows)

    if save_file:
        # save arrays as pickle file with timestamp
        outfile = open(f'windows/picking_windows_{start_Time}.pkl', 'wb')
        pickle.dump(picking_Windows, outfile)
        outfile.close()

        outfile = open(f'windows/picking_windows_untransformed_{start_Time}.pkl',
                       'wb')
        pickle.dump(picking_Windows_Untransformed, outfile)
        outfile.close()

        outfile = open(f'windows/picking_times_{start_Time}.pkl', 'wb')
        pickle.dump(picking_Windows_Times, outfile)
        outfile.close()

    return (picking_Windows, picking_Windows_Untransformed,
            picking_Windows_Times)

# returns Rattlesnake Ridge station locations
def rattlesnake_Ridge_Station_Locations():
    """ Returns a dict of station locations, used by EQTransformer
    downloader.stationListFromMseed to create a station_list.json file"""
    stations = (1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,20,21,22,23,24,25,
                26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,'UGAP3',
                'UGAP5','UGAP6')
    latitudes = (46.5283582,46.5282788,46.5282528,46.5281667,46.5281717,
                 46.5281396,46.5281154,46.5280997,46.5280294,46.5279883,
                 46.527869,46.5280507,46.527738,46.5277918,46.5276182,
                 46.5275217,46.5273933,46.5272481,46.5271363,46.5270128,
                 46.5269864,46.526885,46.5268652,46.5266988,46.5266946,
                 46.5265515,46.526435,46.5264383,46.5263249,46.526291,
                 46.5261454,46.526008,46.5260073,46.5257922,46.5257573,
                 46.5255906,46.525457,46.5253461,46.5253264,46.5251281,
                 46.52764,46.52663,46.52662)
    longitudes = (-120.4667914,-120.4662543,-120.4671993,-120.4663367,
                  -120.4665884,-120.4658735,-120.4668433,-120.4661669,
                  -120.4669906,-120.4660337,-120.4660045,-120.4663792,
                  -120.46595,-120.4662038,-120.4658907,-120.4658828,
                  -120.4657754,-120.465704,-120.4656657,-120.4652239,
                  -120.4656266,-120.466065,-120.4656232,-120.465502,
                  -120.4652094,-120.4655297,-120.466251,-120.4655254,
                  -120.4651244,-120.465529,-120.4655384,-120.4650622,
                  -120.4654893,-120.465437,-120.4649769,-120.4653985,
                  -120.4654196,-120.4650617,-120.4654122,-120.4654584,
                  -120.46564,-120.46553,-120.46637)
    elevations = (435.5141526063508,459.94244464035376,419.17496055551493,
                  456.4002279560358,445.2225853337437,471.96515712383257,
                  433.75775906071993,463.0207557157984,427.2243674889657,
                  466.22453723535386,465.71123190316104,454.23827459858546,
                  459.72188069932287,459.72188069932287,464.2974124921104,
                  462.3340642200634,459.20987810605817,455.81267908165233,
                  452.4536132975136,449.39541344789984,447.7972434217686,
                  444.0736248417953,444.0736248417953,440.47945136171927,
                  441.41787035305947,436.52746672411286,436.52746672411286,
                  433.50566973487213,433.6467649166143,430.2554423791601,
                  425.63420243981653,426.1603861687825,421.53271734479665,
                  416.2025875386629,421.73271284375056,411.9060515678977,
                  407.8851035175373,412.7486362327675,404.519552729108,
                  397.5652740413809,465.2,435.4,435.4)
    location_dict = {}
    for index, station in enumerate(stations):
        location_dict[str(station)] = [longitudes[index], latitudes[index],
                                       elevations[index]]
    return location_dict

# function to find max amplitude of a time series for plotting
def max_Amplitude(timeSeries: Stream or Trace or np.ndarray):
    '''
    (np.ndarray or obspy stream or trace object) -> (float)

    Determines single max value that occurs in a numpy array or trace or in all traces of a
    stream.
    '''
    # for Stream objects
    if isinstance(timeSeries, obspy.core.stream.Stream):
        # creates a list of the max value in each trace
        traceMax = [np.max(np.abs(timeSeries[trace].data)) for trace in
                    range(len(timeSeries))]
        # return max value among all traces
        return np.max(traceMax)

    # for Trace objects
    elif isinstance(timeSeries, obspy.core.trace.Trace):
        traceMax = np.max(np.abs(timeSeries.data))
        # return max value
        return traceMax

    elif isinstance(timeSeries, np.ndarray):
        # returns the max for each row, works with 1D and 2D np.ndarrays
        if len(timeSeries.shape) == 1:  # 1D case
            return np.abs(timeSeries).max()
        elif len(timeSeries.shape) == 2:  # 2D case
            return np.abs(timeSeries).max(1)
        else:
            print("You broke the max_Amplitude function with a np.ndarray "
                  "that is not 1D or 2D ;(")

# build the unet model
def build_unet_model():
    # set model parameters
    drop = 0
    large = 0.5  # large unet, network topology: unet_tools
    epos = 30  # how many epochs?
    std = 0.010  # how long do you want the gaussian STD to be?
    sr = 250  # sample rate

    # build the model save file name
    model_save_file = "unet_logfeat_8399_sn_eps_" + str(epos) + "_sr_" + str(
        sr) + "_std_" + str(std) + ".tf"
    if large:
        fac = large
        model_save_file = "large_" + str(fac) + "_" + model_save_file
    if drop:
        model_save_file = "drop_" + model_save_file

    # build the model
    if drop:
        model = unet.make_large_unet_drop(fac, sr)
    else:
        model = unet.make_large_unet(fac, sr)
    model.load_weights("./" + model_save_file)

    return model

# function to extract pick predictions from unet model
def get_Unet_Picks(picking_Windows, preloaded_model=False):
    """
    Takes in picking_Windows as returned by find_Picking_Windows function,
    and returns arrival time predictions from unet with specified parameters
    (from model file).

    Example:
        detection_Parameters = (120, 60)
        start_Time = UTCDateTime("2018-03-13T01:33:00.0Z")
        end_Time =   UTCDateTime("2018-03-13T01:33:00.48Z")
        filepaths = project_Filepaths("Rattlesnake Ridge", start_Time, end_Time)
        time_Periods = [(start_Time, end_Time)]
        picking_Windows, picking_Windows_Untransformed,
                         picking_Windows_Times = find_Picking_Windows(
                                 filepaths, detection_Parameters, time_Periods)

        # build tensorflow unet model
        model = build_unet_model()
        pick_Predictions = get_Unet_Picks(picking_Windows,preloaded_model=model)
    """
    # check if model has been loaded
    if preloaded_model != False:
        model = preloaded_model

    else:
        model = build_unet_model()

    pick_predictions = model.predict(picking_Windows)

    return pick_predictions

# function to get picks above specified threshold (determined from F1)
def get_Threshold_Picks(pick_Predictions, threshold,
                        picking_Windows_Untransformed, picking_Windows_Times):
    """
    Takes in pick_Predictions, as returned by the get_Unet_Picks function, and
    a threshold to cull the picks to only those above the specified threshold.
    """

    # find the max prediction value in each window
    max_Row_Prediction = np.max(pick_Predictions, axis=1)

    # find all rows with values above the threshold
    above_Threshold_Rows = np.where(max_Row_Prediction > threshold)[0]

    # find max value index in each row with detections above threshold
    index_Of_Max_Prediction = []
    for row in above_Threshold_Rows:
        row_Max_Prediction_Index = np.where(pick_Predictions[row] ==
                                            max_Row_Prediction[row])
        index_Of_Max_Prediction.append(row_Max_Prediction_Index[0][0])

    # cull data to only windows with picks
    picked_Windows = []
    picked_Windows_Times = []
    for row in above_Threshold_Rows:
        picked_Windows.append(picking_Windows_Untransformed[row])
        picked_Windows_Times.append(picking_Windows_Times[row])

    return picked_Windows, picked_Windows_Times, index_Of_Max_Prediction

# function to plot windows and their model predicted picks with minimal info
def plot_Picked_Windows(picked_Windows, index_Of_Max_Prediction,
                        indices=False):
    """Plots normalized windows and the predicted pick for each window, &
    optionally plots data with timestamps. Plot without timestamps in more
    compact. If plotting without timestamps, the indices can be plotted with
    the noise windows by supplying a numpy array containing the indices.


    # FIXME: this function breaks when plotting > 172 windows. Maybe
    # this has to do with matplotlib but I have no proof for this claim
    # and I have not investigated further because life. If you need to
    # plot >172 windows at a time, good luck on your journey.

    Examples
    --------
    # load raw data for windows
    infile = open('picking_windows_untransformed.pkl','rb')
    picking_Windows_Untransformed = pickle.load(infile)
    infile.close()

    # load info associated with windows (times, station, channel)
    infile = open('picking_times.pkl', 'rb')
    picking_Windows_Times = pickle.load(infile)
    infile.close()

    # cull data to only windows with picks
    picked_Windows = []
    picked_Windows_Times = []
    for row in above_Threshold_Rows:
        picked_Windows.append(picking_Windows_Untransformed[row])
        picked_Windows_Times.append(picking_Windows_Times[row])

    # plot 'em utilizing slicing
    plot_Picked_Windows(picked_Windows[150:300], index_Of_Max_Prediction[150:300])
    # breaks when plotting > 172 lines
    """

    # normalize by max value in each window to
    picked_Windows = np.array(picked_Windows)
    maxTraceValue = max_Amplitude(picked_Windows)

    # set the figure size
    figureWidth = 19
    figureHeight = 0.8 * len(picked_Windows)

    # trim number of noise windows if > 172
    if len(picked_Windows) > 172:
        picked_Windows = picked_Windows[0:171]
        index_Of_Max_Prediction = index_Of_Max_Prediction[0:171]

    # if no indices are specified, set them to arbitrary numbers
    if type(indices) == bool:
        indices = list(range(0, len(picked_Windows)))

    fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(figureWidth, figureHeight))
    for index in range(len(picked_Windows)):
        # plot each normalized time series
        data_y_values = (picked_Windows[index] / maxTraceValue[index]) * 0.9\
                        + index + min(indices)
        ax.plot(range(len(picked_Windows[index])), data_y_values, "k-",
                linewidth=0.7)

        # plot line on predicted pick
        pick_y_values = [data_y_values[index_Of_Max_Prediction[index]] -
                         0.5, data_y_values[index_Of_Max_Prediction[index]]
                         + 0.5]
        ax.plot([index_Of_Max_Prediction[index], index_Of_Max_Prediction[
            index]], pick_y_values, "r-", linewidth=1.0)

    ax.set_xlim([0, len(picked_Windows[0]) - 1])
    ax.set_ylim([min(indices) - 1, max(indices) + 1])
    ax.set_title(f"Threshold: {threshold}", fontsize=20)
    # [0, len(picking_Windows_Untransformed) + 1]
    plt.ylabel("Window #")
    plt.xlabel("Sample")
    plt.yticks(indices)
    plt.show()
    #fig.savefig(f"picks_thresh-{threshold}.png", dpi=300)

# removes picks that occur within {duplicate_Threshold} samples of each other
def remove_Duplicate_Picks(picked_Windows, picked_Windows_Times,
                           index_Of_Max_Prediction):
    """
    Removes duplicate picks from picked_Windows, picked_Windows_Times,
    and index_Of_Max_Prediction that occur within duplicate_Threshold
    samples of each other on the same station.
    """
    if len(picked_Windows_Times) > 0:
        # make empty dict for picks
        stas = project_stations("Rattlesnake Ridge", picked_Windows_Times[0][0])
        station_Picks = {}
        for station in stas:
            station_Picks[str(station)] = []

        # station_Picks = {'1': [], '2': [], '3': [], '4': [], '5': [],
        #                  '6': [],
        #                  '7': [], '8': [], '9': [], '10': [], '12': [],
        #                  '13': [],
        #                  '15': [], '16': [], '17': [], '18': [],
        #                  '20': [], '21': [],
        #                  '22': [], '23': [], '25': [], '26': [],
        #                  '27': [], '28': [],
        #                  '30': [], '31': [], '32': [], '33': [],
        #                  '34': [], '35': [],
        #                  '36': [], '37': [], '38': [], '39': [],
        #                  '40': [], '41': [],
        #                  '42': [], 'UGAP3': [], 'UGAP5': [],
        #                  'UGAP6': []}

        duplicate_List = []

        # loop through all picked windows times
        for index in range(0, len(picked_Windows_Times)):
            window_Start, window_End, station, _ = \
                picked_Windows_Times[index]
            pick_Time = window_Start + (
                    index_Of_Max_Prediction[index] / sampling_Rate)

            # if list is empty add pick
            if len(station_Picks[station]) == 0:
                station_Picks[station].append(pick_Time)

            # else add pick to dict if one within duplicate_Threshold doesn't exist
            else:
                duplicate_Flag = False

                for pick in station_Picks[station]:
                    if abs(pick - pick_Time) <= (
                            duplicate_Threshold / sampling_Rate):
                        duplicate_Flag = True
                        duplicate_List.append(index)

                if duplicate_Flag is False:
                    station_Picks[station].append(pick_Time)

        # remove duplicates from picks and metadata
        unique_Picked_Windows = [item for index, item in enumerate(
                                 picked_Windows) if index not in duplicate_List]
        unique_Picked_Windows_Times = [item for index, item in enumerate(
                              picked_Windows_Times) if index not in duplicate_List]
        unique_Index_Of_Max_Prediction = [item for index, item in enumerate(
                           index_Of_Max_Prediction) if index not in duplicate_List]

        return (station_Picks, unique_Picked_Windows, unique_Picked_Windows_Times,
                unique_Index_Of_Max_Prediction)

    else:
        return ({}, [], [], [])


# loads waveforms into Obspy Stream object for plotting picks
def trim_Daily_Waveforms(project_Name: str, start_Time, end_Time, channels:
                         list, write_File=False):
    '''
    loads project data into an Obspy Stream object. By default this will 
    grab the entire day. If start_Time and end_Time are specified the Stream 
    object will be trimmed to span that period. start_Time and end_Time are 
    UTCDateTime objects.

    Example: for all stations in one stream with distance hack, for picking
    start_Time = UTCDateTime("2018-03-13T01:33:00.1Z")
    end_Time =   UTCDateTime("2018-03-13T01:33:30.0Z")
    project_Name = "Rattlesnake Ridge"
    channels = ['DP1', 'EHN']
    stream = trim_Daily_Waveforms(project_Name, start_Time, end_Time, 
    channels, write_File=False)

    '''
    project_Aliases = {"Rattlesnake Ridge": "RR"}

    if project_Name == "Rattlesnake Ridge":
        # build filepath list based on dates, station type, and channels
        node = ['DP1', 'DP2', 'DPZ']  # nodal seismometer channels
        ugap = ['EHN', 'EHE', 'EHZ']
        stas = project_stations("Rattlesnake Ridge", start_Time)
        stations_channels = {}
        for station in stas:
            if isinstance(station, int):
                stations_channels[str(station)] = node
            elif isinstance(station, str):
                stations_channels[station] = ugap

        # stations_channels = {'1': node, '2': node, '3': node, '5': node,
        #                      '4': node, '6': node, '7': node, '8': node,
        #                      '13': node, '9': node, '10': node, '12': node,
        #                      '15': node, 'UGAP3': ugap, '16': node, '17': node,
        #                      '18': node, '20': node, '21': node, '22': node,
        #                      '23': node, '25': node, '26': node, '27': node,
        #                      'UGAP5': ugap, 'UGAP6': ugap, '28': node,
        #                      '30': node, '31': node, '32': node, '33': node,
        #                      '34': node, '35': node, '36': node, '37': node,
        #                      '38': node, '39': node, '40': node, '41': node,
        #                      '42': node}

        # to view stations in distance along scarp for picking in Snuffler
        station_distance_hack = {station: index for index, station in
                                 enumerate(stations_channels)}

        filepaths = []
        for station in stations_channels:
            for channel in stations_channels[station]:
                if channel in channels:  # is this a channel we specified?
                    # EXTERNAL HD LOCATION
                    # filepath = f"/Volumes/newton_6TB/RR_MSEED/5A.{station}.." \
                    #            f"{channel}.{start_Time.year}-" \
                    #            f"{start_Time.month:02}-{start_Time.day:02}T00.00.00.ms"
                    # LOCAL LOCATION
                    filepath = f"../shared/Rattlesnake_2018/5A.{station}.." \
                               f"{channel}.{start_Time.year}-" \
                               f"{start_Time.month:02}-{start_Time.day:02}T00.00.00.ms"
                    filepaths.append(filepath)

        obspyStream = Stream()
        for filepath_idx in range(len(filepaths)):
            obspyStream += read(filepaths[filepath_idx]).merge(method=1,
                                                               fill_value=0)
            # station-distance hack for picking, assign number to network
            hack = station_distance_hack[obspyStream[
                filepath_idx].stats.station]
            obspyStream[filepath_idx].stats.network = f'{hack:02}'

    # make sure all traces have the same sampling rate (and thus number of
    # samples and length) to avoid bugs in other programs, e.g. Snuffler
    # slows down with variable sampling rates
    interpolate = False
    for index, trace in enumerate(obspyStream):
        # the sampling rate of the first trace is assumed to be correct
        if trace.stats.sampling_rate != obspyStream[0].stats.sampling_rate:
            # print(f"Trace {index} has a different sampling rate. ")
            # print(f"Station {trace.stats.station}, Channel "
            #       f"{trace.stats.channel}, Start: "
            #       f"{trace.stats.starttime}, End: {trace.stats.endtime}")
            # raise the flag
            interpolate = True
    if interpolate:
        print("Interpolating...")
        # interpolate to correct sampling rate and trim to correct time period
        sampling_Rate = obspyStream[0].stats.sampling_rate
        obspyStream = obspyStream.trim(start_Time - sampling_Rate, end_Time
                                       + sampling_Rate)
        npts = int((end_Time - start_Time) * sampling_Rate)
        obspyStream.interpolate(sampling_Rate, method="lanczos",
                                starttime=start_Time, npts=npts, a=30)
        # obspy.signal.interpolation.plot_lanczos_windows(a=30)
    else:
        # trim to specified time period
        obspyStream = obspyStream.trim(start_Time, end_Time)

    if write_File:
        # format filename and save Stream as miniseed file
        start_Time_Stamp = str(obspyStream[0].stats.starttime)[
                           :19].replace(":", ".")
        # writes to project path
        obspyStream.write(f"/Users/human/Dropbox/Programs/unet/waveforms/"
                          f"{start_Time_Stamp}.ms", format="MSEED")

    return obspyStream

# plots a single window of time and the unet picks from that window
def plot_Picked_Window(window_times, cluster_picked_windows,
                 cluster_picked_windows_times, cluster_index_of_max_prediction):
    """
    Plots time series' and unet picks for a single window of time.

    window_times = (UTCDateTime("2018-03-13T01:33:00.0Z"),
                    UTCDateTime("2018-03-13T01:33:05.0Z"))
    plot_Picked_Window(window_times, picked_Windows,
                       picked_Windows_Times, index_Of_Max_Prediction)
    """
    figureWidth = 4
    figureHeight = 14
    fig, ax = plt.subplots(figsize=(figureWidth, figureHeight))

    # time period to plot (0.48 second window)
    plot_start_time, plot_end_time = window_times

    # get stream of data for plotting
    channels = ['DP1', 'EHN']
    waveforms = trim_Daily_Waveforms("Rattlesnake Ridge", plot_start_time,
                                     plot_end_time, channels, write_File=False)

    # find and remove duplicate predictions
    cluster_station_picks, cluster_picked_windows, \
    cluster_picked_windows_times, cluster_index_of_max_prediction = \
            remove_Duplicate_Picks(cluster_picked_windows,
                 cluster_picked_windows_times, cluster_index_of_max_prediction)

    # get the station location dictionary
    location_dict = rattlesnake_Ridge_Station_Locations()

    cluster_pick_array = []
    cluster_pick_station_order = []  # station distance along scarp
    cluster_pick_times = []

    # dict to convert station to station distance proxy. order matters.
    stas = project_stations("Rattlesnake Ridge", window_times[0])
    stations = {}
    for station in stas:
        stations[str(station)] = []
    # stations = {'1': [], '2': [], '3': [], '5': [], '4': [], '6': [], '7': [],
    #             '8': [], '13': [], '9': [], '10': [], '12': [], '15': [],
    #             'UGAP3': [], '16': [], '17': [], '18': [], '20': [], '21': [],
    #             '22': [], '23': [], '25': [], '26': [], '27': [], 'UGAP5': [],
    #             'UGAP6': [], '28': [], '30': [], '31': [], '32': [], '33': [],
    #             '34': [], '35': [], '36': [], '37': [], '38': [], '39': [],
    #             '40': [], '41': [], '42': []}

    # station "distance" along scarp (it's actually station order along scarp)
    station_distance = {station: index for index, station in
                        enumerate(stations)}

    # build numpy array of longitude, latitude, elevation, arrival time for each
    # detected arrival time
    for station in cluster_station_picks.keys():
        for pick_time in cluster_station_picks[station]:
            # get station location
            pick_location_and_time = location_dict[station].copy()

            # then get pick time and convert from UTCDateTime object to number
            pick_location_and_time.append(pick_time.matplotlib_date)

            # append to major list
            cluster_pick_array.append(pick_location_and_time)

            # save station and time for future processing (plotting clusters)
            cluster_pick_station_order.append(station_distance[station])
            cluster_pick_times.append(pick_time.matplotlib_date)

    # convert to np.array for bool indexing
    cluster_pick_station_order = np.array(cluster_pick_station_order)
    cluster_pick_times = np.array(cluster_pick_times)

    # # calculate median time of picks for each cluster
    # cluster_Medians = {}
    # cluster_Means = {}
    # for label in unique_labels:
    #     member_Indices = np.where(labels == label)[0]
    #     cluster_Medians[label] = np.median(pick_Times[member_Indices])
    #     cluster_Means[label] = np.mean(pick_Times[member_Indices])

    ax.plot_date(cluster_pick_times, cluster_pick_station_order, 'o',
                  markerfacecolor='r', markeredgecolor='k', markersize=9)
    ax.plot_date(cluster_pick_times, cluster_pick_station_order, '|',
                  markerfacecolor='k', markeredgewidth=0.2,
                  markeredgecolor='k', markersize=9)

    # # plot cluster median
    # y2 = [pick_Station_Order[class_member_mask].min(),
    #       pick_Station_Order[class_member_mask].max()]
    # x2 = [cluster_Median, cluster_Median]
    # plt.plot_date(x2, y2, fmt='-', linewidth=0.4, color='grey')
    #
    # # plot cluster mean
    # x3 = [cluster_Mean, cluster_Mean]
    # plt.plot_date(x3, y2, fmt='-', linewidth=0.4,
    #               color='green')

    # set axes attributes on first subplot
    ax.set_yticks(np.arange(0, len(station_distance)))
    ax.set_yticklabels(station_distance.keys())
    ax.set_ylabel('Station')
    # ax_DBSCAN0.set_xticks(np.arange(0, 1.1, 0.5))
    ax.set_xlim([plot_start_time.matplotlib_date,
                         plot_end_time.matplotlib_date])
    # ax_DBSCAN0.set_xticklabels(np.arange(0, 1.1, 0.2))
    # ax_DBSCAN0.set_xlabel('Seconds')

    myFmt = DateFormatter("%S.%f")  # "%H:%M:%S.f"
    ax.xaxis.set_major_formatter(myFmt)
    locator_x = AutoDateLocator(minticks=2, maxticks=3)
    ax.xaxis.set_major_locator(locator_x)
    ax.set_ylim((-1, len(station_distance)))
    # ax_DBSCAN0.set_xlim((0, 1))
    # fig.tight_layout()

    # reverse y axis
    ax.set_ylim(ax.get_ylim()[::-1])

    # plot waveforms for each station
    # normalize by max value in each window
    maxTraceValue = max_Amplitude(waveforms)
    for trace in range(len(waveforms)):
        ax.plot_date(waveforms[trace].times(type="matplotlib"),
                      (waveforms[trace].data / maxTraceValue) * 1.25 +
                      trace, fmt="k-", linewidth=0.4)

    ax.set_xlabel('Time (seconds)')
    ax.set_title('Repicking around cluster median', fontweight="bold",
                 y=0.9999)
    fig.tight_layout()
    # fig.savefig(f'repicking_Results.pdf', dpi=600)
    plt.show()

    return None

# makes new picks for each cluster based on the cluster median
def repick_Around_Cluster_Medians(labels, pick_times, filepaths,
                                  detection_Parameters, sampling_Rate,
                                  threshold, station_distance,
                                  location_dict, model):
    """
    Determines the median of each cluster and makes new picks surrounding
    that cluster median. Considers three windows for unet repicking: -0.20
    second offset, 0.0 second offset, and +0.20 second offset. These three
    windows are used for repicking because repicking a single window around
    the cluster median does not always capture signals in that window.
    """
    # build set of cluster labels
    unique_labels = set(labels)

    # convert to np.array for indexing
    pick_times = np.array(pick_times)

    # calculate median time of picks for each cluster
    cluster_medians = {}
    for label in unique_labels:
        member_indices = np.where(labels == label)[0]
        cluster_medians[label] = np.median(pick_times[member_indices])

    # # # # # # # # # for testing # # # # # # # #
    # UTC_cluster_meds = {}
    # for key in cluster_medians.keys():
    #     UTC_cluster_meds[key] = UTCDateTime(num2date(cluster_medians[key]))
    #     print(f"{key} - {UTC_cluster_meds[key]}")
    # # # # # # # # # # # # # # # # # # # # # # #

    # initialize lists to rebuild
    new_labels = []
    new_pick_times = []
    new_pick_station_order = []

    # loop through clusters
    for cluster in unique_labels:

        # ignore noise cluster with label of -1
        if cluster != -1:

            # get median for current cluster
            cluster_median = cluster_medians[cluster]

            # build list of time windows for wobbly repicking around cluster
            # median consisting of original window, and two windows offset
            # by plus and minus 0.20 seconds
            time_periods = []

            window_times = (UTCDateTime(num2date(cluster_median)) -
                           (detection_Parameters[0] / sampling_Rate) / 2,
                            UTCDateTime(num2date(cluster_median)) +
                            (detection_Parameters[0] / sampling_Rate) / 2)

            # append window offset by -0.20 seconds
            time_periods.append((window_times[0] - 0.20, window_times[1] -0.2))
            # append window without offset
            time_periods.append(window_times)
            # append window offset by +0.20 seconds
            time_periods.append((window_times[0] + 0.20, window_times[1] +0.2))

            # get picking windows for unet input
            cluster_picking_windows, cluster_picking_windows_untransformed, \
            cluster_picking_windows_times = find_Picking_Windows(filepaths,
                                             detection_Parameters,time_periods)

            # get phase arrival time predictions from pretrained unet model
            cluster_pick_predictions = get_Unet_Picks(
                cluster_picking_windows, preloaded_model=model)

            # get only picks above specified threshold (determined from F1)
            cluster_picked_windows, cluster_picked_windows_times, \
            cluster_index_of_max_prediction = get_Threshold_Picks(
                cluster_pick_predictions, threshold,
                cluster_picking_windows_untransformed,
                cluster_picking_windows_times)

            # find and remove duplicate picks
            cluster_station_picks, cluster_picked_windows, \
            cluster_picked_windows_times, cluster_index_of_max_prediction = \
                remove_Duplicate_Picks(cluster_picked_windows,
                                       cluster_picked_windows_times,
                                       cluster_index_of_max_prediction)

            # # plot new picks [for testing]
            # plot_Picked_Window(window_times, cluster_picked_windows,
            #                    cluster_picked_windows_times,
            #                    cluster_index_of_max_prediction)

            # append labels, pick times, and pick station order to new lists
            for station in cluster_station_picks.keys():
                for pick_time in cluster_station_picks[station]:
                    # get station location
                    pick_location_and_time = location_dict[station].copy()

                    # then get pick time and convert from UTCDateTime object to number
                    pick_location_and_time.append(pick_time.matplotlib_date)

                    # save station, time, and label
                    new_pick_station_order.append(station_distance[station])
                    new_pick_times.append(pick_time.matplotlib_date)
                    new_labels.append(cluster)

    return new_pick_station_order, new_pick_times, new_labels

# builds pick dictionary that is used by other functions
def build_Pick_Dict(pick_times, pick_station_order, reverse_station_distance,
                    duplicate_threshold, sampling_rate):
    """
    Builds a dictionary containing times of picks on each station, excluding
    duplicates.
    """
    station_picks = {}
    if len(pick_times) > 0:
        # make empty dict for picks
        stas = project_stations("Rattlesnake Ridge", UTCDateTime(num2date(
                                pick_times[0])))

        for station in stas:
            station_picks[str(station)] = []
        # station_picks = {'1': [], '2': [], '3': [], '4': [], '5': [], '6': [],
        #                  '7': [], '8': [], '9': [], '10': [], '12': [], '13': [],
        #                  '15': [], '16': [], '17': [], '18': [], '20': [],
        #                  '21': [], '22': [], '23': [], '25': [], '26': [],
        #                  '27': [], '28': [], '30': [], '31': [], '32': [],
        #                  '33': [], '34': [], '35': [], '36': [], '37': [],
        #                  '38': [], '39': [], '40': [], '41': [], '42': [],
        #                  'UGAP3': [], 'UGAP5': [], 'UGAP6': []}

        #duplicate_list = []

        # loop through all pick times
        for pick_index, station in enumerate(pick_station_order):
            # define station based on reverse_station_distance lookup dict
            station = reverse_station_distance[station]
            # if list is empty add pick
            if len(station_picks[station]) == 0:
                station_picks[station].append(UTCDateTime(num2date(pick_times[
                                                                     pick_index])))

            # else add pick to dict if one within duplicate_Threshold doesn't exist
            else:
                duplicate_flag = False

                for pick in station_picks[station]:
                    if abs(pick - UTCDateTime(num2date(pick_times[pick_index])))\
                     <= (duplicate_threshold / sampling_rate):
                        duplicate_flag = True
                        #duplicate_list.append(pick_index)

                if duplicate_flag is False:
                    station_picks[station].append(UTCDateTime(num2date(
                                                          pick_times[pick_index])))

    return station_picks

# builds lists of pick data that are used by other functions
def build_Pick_Data(station_Picks, location_Dict, station_Distance):
    """
    Builds pick_Array, pick_Station_Order, and pick_Times, which are used by
    other functions to process and manipulate picks.
    """
    pick_Array = []
    pick_Station_Order = []  # station distance along scarp
    pick_Times = []
    # build numpy array of longitude, latitude, elevation, arrival time for each
    # detected arrival time
    for station in station_Picks.keys():
        for pick_Time in station_Picks[station]:
            # get station location
            pick_Location_And_Time = location_Dict[station].copy()

            # then get pick time and convert from UTCDateTime object to number
            pick_Location_And_Time.append(pick_Time.matplotlib_date)

            # append to major list
            pick_Array.append(pick_Location_And_Time)

            # save station and time for future processing (plotting clusters)
            pick_Station_Order.append(station_Distance[station])
            pick_Times.append(pick_Time.matplotlib_date)

    return pick_Array, pick_Station_Order, pick_Times

# clusters picks using the specified algorithm and parameters
def cluster_Picks(pick_Array, clustering_Algo, scale_Time, PCA,
                  PCA_Components):
    """
    # TODO: add docstring
    """
    # standardize data
    pick_Array = np.asarray(pick_Array)
    pick_Array = np.nan_to_num(pick_Array) # control for NaNs
    pick_Array = (pick_Array - np.mean(pick_Array, axis=0)) / np.std(
                  pick_Array, axis=0)
    pick_Array = np.nan_to_num(pick_Array) # control for NaNs

    # time scaling for association clustering tuning
    if scale_Time:
        pick_Array[:, 3] *= scale_Factor

    # perform pca on pick data, or not
    if PCA:
        # check if number of PCA components is specified, else use default
        if PCA_Components:
            pca = decomposition.PCA(n_components=PCA_Components)
        else:
            pca = decomposition.PCA()
        pca.fit(pick_Array)
        pick_Array_PCA = pca.transform(pick_Array)

        # if PCA is specified, use PCA transformed data as clustering data
        data_To_Cluster = pick_Array_PCA

        # print(f"PCA - Explained variance ratios for each feature:\
        #         \n{pca.explained_variance_ratio_}")

    else:
        # no PCA, cluster standardized data
        data_To_Cluster = pick_Array

    # cluster the data
    if clustering_Algo == "DBSCAN":
        # run DBSCAN
        db = DBSCAN(eps=DBSCAN_Distance, min_samples=DBSCAN_Min_Samples).fit(
            data_To_Cluster)

        labels = db.labels_

    # elif clustering_Algo == "OPTICS":
    # TODO
    # else:
    # TODO

    return labels

# determines median time of each cluster and member indices
def get_cluster_stats(labels, pick_times):
    """
    Returns a set of unique labels within labels, and a dict of the medians
    for each cluster.
    """
    # build set of cluster labels
    unique_labels_set = set(labels)

    # make sure pick_times is a np.array for indexing
    pick_times = np.array(pick_times)

    # calculate median time of picks for each cluster
    cluster_medians = {}
    for label in unique_labels_set:
        member_indices = np.where(labels == label)[0]
        cluster_medians[label] = np.median(pick_times[member_indices])

    return unique_labels_set, cluster_medians

# association helper function to process clusters of picks
def process_Clusters(labels, pick_Times, pick_Station_Order,
                     min_Quakes_Per_Cluster, temporal_Threshold,
                     spatial_Threshold, merge_clusters=False,
                     process_noise=False):
    """
    Processes clusters of picks in the following way:
        - disolves clusters with < min_Quakes_Per_Event picks
        - merges clusters within XXX seconds and deletes duplicate picks

    """
    # make sure pick_times and pick_Station_Order are np.arrays for indexing
    pick_Times = np.array(pick_Times)
    pick_Station_Order = np.array(pick_Station_Order)

    # get set of unique labels
    unique_labels, cluster_medians = get_cluster_stats(labels, pick_Times)

    # # # # # # # # # for testing # # # # # # # #
    # UTC_cluster_meds = {}
    # for key in cluster_medians.keys():
    #     UTC_cluster_meds[key] = UTCDateTime(num2date(cluster_medians[key]))
    #     print(f"{key} - {UTC_cluster_meds[key]}")
    #
    # # close clusters, 0.04 s apart for period_Index = 0
    # UTC_cluster_meds[0] - UTC_cluster_meds[8]
    # # # # # # # # # # # # # # # # # # # # # # #

    # loop through clusters
    for cluster in unique_labels:

        # NOISE CASE: if cluster is the noise cluster (label == -1)
        if cluster == -1:
            # skip noise cluster and process it last (outside of for loop)
            continue

        # SIGNAL CASE: the cluster is not considered noise, so process picks
        else:
            # build list of indices of picks in current cluster
            member_indices = np.where(labels == cluster)[0]

            # check for minimum number of picks in cluster
            if len(member_indices) < min_Quakes_Per_Cluster:
                # set labels to noise (-1) if minimum number isn't met
                labels[member_indices] = -1

                # don't continue processing, skip to next cluster
                continue

            # cluster has sufficient number of picks to continue processing
            else:
                # rebuild cluster medians with each iteration
                _ , cluster_medians = get_cluster_stats(labels, pick_Times)

                # get median for current cluster
                cluster_median = UTCDateTime(num2date(cluster_medians[
                                                          cluster]))

                # build list of stations in cluster
                unique_cluster_stations = set(pick_Station_Order[
                                                  member_indices])
                all_cluster_stations = list(pick_Station_Order[member_indices])

                # CASE 0: multiple picks on a single station
                duplicates = defaultdict(list)
                for duplicate_index, duplicate_station in enumerate(
                                                         all_cluster_stations):
                    duplicates[duplicate_station].append(duplicate_index)
                # transform to dict of duplicate stations : indices
                duplicates = {dup_station : dup_indices for dup_station,
                              dup_indices in duplicates.items() if
                              len(dup_indices) > 1}

                # loop through all duplicates and calculate distance from median
                for dup_station in duplicates.keys():
                    # initialize list to store distance from median
                    median_distances = []
                    for index_index in duplicates[dup_station]:
                        # get pick time as UTCDateTime object
                        pick_time = UTCDateTime(num2date(pick_Times[
                                                 member_indices[index_index]]))
                        # calculate distance from cluster median
                        # dist = abs(pick_time - cluster_median) # closest to median
                        dist = pick_time - cluster_median # first pick
                        # append distance from cluster median
                        median_distances.append(dist)

                    # find minimum distance and its index
                    min_dist = min(median_distances)
                    min_index = median_distances.index(min_dist)

                    # keep first pick, (or closest to med, defined above) else noise
                    for place_in_list, index_index in enumerate(duplicates[
                                                                 dup_station]):
                        if place_in_list == min_index:
                            continue
                        else:
                            # change label to noise and remove as member
                            labels[member_indices[index_index]] = -1

                # rebuild member indices
                member_indices = np.where(labels == cluster)[0]

                # check for minimum number of picks in cluster
                if len(member_indices) < min_Quakes_Per_Cluster:
                    # set labels to noise (-1) if minimum number isn't met
                    labels[member_indices] = -1

                    # don't continue processing, skip to next cluster
                    continue

                # there are enough picks, continue processing cluster
                else:
                    # rebuild cluster medians after deleting duplicates
                    _, cluster_medians = get_cluster_stats(labels,pick_Times)

                    # re-get median for current cluster after deleting duplicates
                    cluster_median = UTCDateTime(num2date(cluster_medians[cluster]))

                    # rebuild list of stations in cluster after deleting duplicates
                    unique_cluster_stations = set(pick_Station_Order[member_indices])
                    all_cluster_stations = list(pick_Station_Order[member_indices])

                    # CASE 1: misfit picks in cluster
                    # loop through each pick in the cluster to check fit within cluster
                    for index, pick in enumerate(pick_Times[member_indices]):
                        cluster_stations = list(unique_cluster_stations.copy())

                        # get pick station
                        pick_Station = pick_Station_Order[member_indices][index]

                        # find closest station (excluding station pick occurs on)
                        cluster_stations.remove(pick_Station)
                        station_Differences = [abs(pick_Station - cluster_Station) for
                                               cluster_Station in cluster_stations]
                        min_Difference_Index = min(range(len(station_Differences)),
                                                   key=station_Differences.__getitem__)

                        # drop all picks outside of specified time threshold from median
                        # cluster pick time & specified space threshold
                        pick = UTCDateTime(num2date(pick))
                        if (abs(pick - cluster_median) > temporal_Threshold) \
                                or (abs(pick_Station - cluster_stations[
                                      min_Difference_Index]) > spatial_Threshold):
                            # remove pick from cluster by changing label at index to noise
                            labels[member_indices[index]] = -1

                # CASE 2: all picks have been removed from cluster
                # check if cluster still has members
                member_indices = np.where(labels == cluster)[0]
                # condition for empty cluster
                # if len(member_indices) == 0:
                #     print(f"cluster {cluster} disolved")

                # CASE 3: find unpicked arrivals
                # check for missed arrivals on stations without picks
                # (STA/LTA? and fit) or shift window start time back by
                # some amount and repick with unet

                # CASE 4: cluster should be merged with another cluster or
                # deleted. check if cluster picks should be merged with
                # another cluster with a time, space, and minimum member
                # condition. If median of a cluster is within thresh,
                # merge clusters. if some points are outside space or time
                # thresh split or generate new cluster (if enough points
                # outside to form new cluster) outside_thresh counter. Check
                # cluster med is outside of some threshold from previous
                # cluster median

    # CASE: merge overlapping clusters
    # check if any clusters should be merged with time and space condition
    if merge_clusters:
        unique_labels, cluster_medians = get_cluster_stats(labels, pick_Times)
        unique_labels.remove(-1) # do not consider the noise cluster for merging
        cluster_medians.pop(-1, None) # do not consider the noise cluster
        merge_dict = {} # merge key into value
        # loop through clusters and get ready for a lot of conditionals
        for cluster in unique_labels:
            # get median for current cluster
            current_cluster_median = UTCDateTime(num2date(cluster_medians[cluster]))
            # loop through all cluster medians
            for other_cluster in cluster_medians.keys():
                # exclude comparing a cluster to itself
                if other_cluster != cluster:
                    other_cluster_median = UTCDateTime(num2date(cluster_medians[
                                                                   other_cluster]))
                    # check that neither cluster is already in merge dict
                    if (other_cluster not in merge_dict.keys()) and (cluster not
                                                             in merge_dict.keys()):
                        # compare medians with temporal threshold
                        if abs(current_cluster_median - other_cluster_median) <= \
                                temporal_Threshold:
                            # build information to compare spatial condition
                            cluster_member_indices = np.where(labels == cluster)[0]
                            other_cluster_member_indices = np.where(labels ==
                                                                  other_cluster)[0]
                            cluster_pick_stations = set(pick_Station_Order[
                                                        cluster_member_indices])
                            other_cluster_pick_stations = set(pick_Station_Order[
                                                     other_cluster_member_indices])
                            # if there is no intersection between cluster stations
                            # check the distance between all stations in each set
                            if cluster_pick_stations.isdisjoint(
                                                      other_cluster_pick_stations):
                                # convert back to lists to be subscriptable
                                cluster_pick_stations = list(cluster_pick_stations)
                                other_cluster_pick_stations = list(
                                                       other_cluster_pick_stations)

                                # loop through each station in current cluster
                                for pick_station in cluster_pick_stations:
                                    station_differences = [
                                        abs(pick_station - other_cluster_station)
                                        for other_cluster_station in
                                        other_cluster_pick_stations]
                                    min_difference_index = min(
                                        range(len(station_differences)),
                                        key=station_differences.__getitem__)

                                    # check if spatial threshold is met to merge
                                    if abs(pick_station -
                                           other_cluster_pick_stations[
                                           min_difference_index]) <= \
                                           spatial_Threshold:

                                        # merge condition met, add entry to
                                        # merge_dict if it does not already exist
                                        if (other_cluster not in
                                           merge_dict.keys()) and (cluster not
                                           in merge_dict.keys()):

                                            merge_dict[cluster] = other_cluster

                            # if the sets intersect merge the clusters
                            else:
                                # there are picks on the same station, merge clusts
                                merge_dict[cluster] = other_cluster

        # if there are clusters to merge, merge dict key into value
        if len(merge_dict.keys()) != 0:
            # print(f"merge dict: {merge_dict}")
            for cluster in merge_dict.keys():
                # print(f"Merging cluster {cluster} into cluster {other_cluster}")
                other_cluster = merge_dict[cluster]
                cluster_member_indices = np.where(labels == cluster)[0]
                # change label from cluster to other cluster
                labels[cluster_member_indices] = other_cluster

    # CASE: event picks assigned as noise
    # see if picks labeled as noise should belong to a cluster
    if process_noise:
        # process noise cluster last
        cluster = -1
        member_indices = np.where(labels == cluster)[0]
        # rebuild cluster medians
        _, cluster_medians = get_cluster_stats(labels, pick_Times)

        # loop through each pick that is considered noise
        for index, pick in enumerate(pick_Times[member_indices]):
            # convert pick to UTCDateTime
            pick = UTCDateTime(num2date(pick))

            # loop through each cluster
            for cluster_key in cluster_medians.keys():
                # get median for current cluster
                cluster_median = UTCDateTime(num2date(cluster_medians[cluster_key]))

                # get pick station
                noise_pick_station = pick_Station_Order[member_indices][index]
                # get array of stations with picks in the current cluster
                cluster_member_indices = np.where(labels == cluster_key)[0]
                cluster_pick_stations = list(set(pick_Station_Order[
                                                      cluster_member_indices]))

                # find closest station
                station_differences = [abs(noise_pick_station -
                                       cluster_pick_station) for
                                 cluster_pick_station in cluster_pick_stations]
                # guard against empty station_differences
                if len(station_differences) > 0:
                    min_difference_index = min(range(len(station_differences)),
                                               key=station_differences.__getitem__)

                    # check if noise pick is within temporal and spatial threshold of
                    # cluster median
                    if (abs(pick - cluster_median) <= temporal_Threshold) and (
                        abs(noise_pick_station - cluster_pick_stations[
                                      min_difference_index]) <= \
                        spatial_Threshold):
                        # check if there is already a pick on this station for this cluster
                        if noise_pick_station not in cluster_pick_stations:
                            # change label from noise to cluster number
                            labels[member_indices[index]] = cluster_key

                        # else do nothing
                        else:
                            continue

                    # if pick is outside of temporal threshold from cluster med. do nothing
                    else:
                        continue

    return labels

# associates signal arrival times via manual decision tree
def associate_Picks(clustering_Algo, station_Picks, scale_Time, PCA,
                    PCA_Components, duplicate_Threshold, temporal_Threshold,
                    spatial_Threshold, min_Quakes_Per_Cluster, model):
    """
    Forms clusters from picks then alters them based on a manual decision
    tree to associate phase picks into events and associated picks.
    """
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # preprocess the data for clustering # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # get the station location dictionary
    location_Dict = rattlesnake_Ridge_Station_Locations()

    # dict to convert station to station distance proxy. order matters
    for station in station_Picks.keys():
        if len(station_Picks[station]) > 0:
            break
    stas = project_stations("Rattlesnake Ridge", station_Picks[station][0])
    stations = {}
    for station in stas:
        stations[str(station)] = []
    # stations = {'1': [], '2': [], '3': [], '5': [], '4': [], '6': [], '7': [],
    #             '8': [], '13': [], '9': [], '10': [], '12': [], '15': [],
    #             'UGAP3': [], '16': [], '17': [], '18': [], '20': [], '21': [],
    #             '22': [], '23': [], '25': [], '26': [], '27': [], 'UGAP5': [],
    #             'UGAP6': [], '28': [], '30': [], '31': [], '32': [], '33': [],
    #             '34': [], '35': [], '36': [], '37': [], '38': [], '39': [],
    #             '40': [], '41': [], '42': []}

    # station "distance" along scarp (it's actually station order along scarp)
    station_Distance = {station: index for index, station in
                        enumerate(stations)}
    reverse_station_distance = {index: station for index, station in
                                enumerate(stations)}

    # build pick data structures used by other functions
    pick_Array, pick_Station_Order, pick_Times = build_Pick_Data(
                                station_Picks, location_Dict, station_Distance)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # cluster picks # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # only attempt to associate if there are picks in pick array
    if len(pick_Array) > 0:

        labels = cluster_Picks(pick_Array, clustering_Algo, scale_Time, PCA,
                               PCA_Components)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # repick via unet around cluster medians # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        pick_Station_Order, pick_Times, labels = repick_Around_Cluster_Medians(
                                        labels, pick_Times, filepaths,
                                        detection_Parameters, sampling_Rate,
                                        threshold, station_Distance,
                                        location_Dict, model)

        # rebuild dictionary of picks and remove duplicates
        station_Picks = build_Pick_Dict(pick_Times, pick_Station_Order,
                                        reverse_station_distance,
                                        duplicate_Threshold, sampling_Rate)
        # rebuild pick data for further processing
        pick_Array, pick_Station_Order, pick_Times = build_Pick_Data(
                                    station_Picks, location_Dict, station_Distance)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # recluster repicked picks # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # only continue association if there are picks in pick array
        if len(pick_Array) > 0:

            labels = cluster_Picks(pick_Array, clustering_Algo, scale_Time, PCA,
                                   PCA_Components)

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # # # # # # # # # # process clusters # # # # # # # # # # # # # # #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

            # first process clusters and noise with merging
            labels = process_Clusters(labels, pick_Times, pick_Station_Order,
                                      min_Quakes_Per_Cluster, temporal_Threshold,
                                      spatial_Threshold, merge_clusters=True,
                                      process_noise=True)

            # process clusters again to clean up cluster changes, without noise & merge
            labels = process_Clusters(labels, pick_Times, pick_Station_Order,
                                      min_Quakes_Per_Cluster, temporal_Threshold,
                                      spatial_Threshold, merge_clusters=False,
                                      process_noise=False)

        # generate blank labels object if there are no picks
        else:
            labels = []

    # generate blank labels object if there are no picks
    else:
        labels = []

    return station_Distance, reverse_station_distance, pick_Station_Order, \
           pick_Times, labels

# saves association results to a snuffler-type marker file
# i.e. https://pyrocko.org/docs/current/formats/snuffler_markers.html
def save_associations(event_filename, labels, pick_Times,
                      pick_Station_Order, reverse_station_distance):
    """Saves association results to snuffler-type marker file
    https://github.com/pyrocko/pyrocko/blob/4971d5e5ac3c4ac4b75992452a934e01a784ea8d/src/gui/marker.py
    """
    # initialize list to store content to write to file
    content_list = []

    # make sure pick_times and pick_Station_Order are np.arrays for indexing
    pick_Times = np.array(pick_Times)
    pick_Station_Order = np.array(pick_Station_Order)

    # get set of cluster labels and medians
    unique_labels, cluster_medians = get_cluster_stats(labels, pick_Times)
    # remove noise cluster
    if -1 in unique_labels:
        unique_labels.remove(-1)

    # loop through each cluster label except noise cluster
    for cluster in unique_labels:
        # define the event time
        event_time = num2date(cluster_medians[cluster]).strftime('%Y-%m-%d '
                                                                 '%H:%M:%S.%f')
        # chop microseconds to five digits per snuffler format
        event_time = event_time[:-1]
        # generate event hash
        event_hash = str(base64.urlsafe_b64encode(hashlib.sha1(
            event_time.encode('utf8')).digest()).decode('ascii'))

        # build the event line for the file
        event_line = f"event: {event_time}  0 {event_hash}          0.0          0.0 None         None None  Event None\n"
        # append event line to content list
        content_list.append(event_line)

        # build list of indices of picks in current cluster
        member_indices = np.where(labels == cluster)[0]

        # loop through each pick in the cluster
        for index, pick in enumerate(pick_Times[member_indices]):
            # get pick station and station distance order hack
            pick_station_hack = pick_Station_Order[member_indices][index]
            pick_station = reverse_station_distance[pick_station_hack]
            # get pick channel
            if pick_station[0] == "U":
                # UGAP channel is EHN
                pick_channel = "EHN"
            else:
                # node channel is DP1
                pick_channel = "DP1"

            # get pick time
            pick_time = num2date(pick).strftime('%Y-%m-%d %H:%M:%S.%f')
            # chop microseconds to five digits per snuffler format
            pick_time = pick_time[:-1]

            # build station/channel line
            pick_station_channel = f"{pick_station_hack}.{pick_station}.." \
                                   f"{pick_channel}"
            # build the pick line for the file. P picks only, no polarity.
            pick_line = f"phase: {pick_time}  0 {pick_station_channel: <16}" \
                        f"{event_hash} {event_time[:10]}  {event_time[10:]} " \
                        f"P        None False\n"
            # append pick line to content list
            content_list.append(pick_line)

    # append contents to file
    f = open(event_filename, "a")
    f.writelines(content_list)
    f.close()

    return None

# plots association results for the specified time period
def build_association_figure(figure_Columns, period_Index, labels,
                             pick_Station_Order, pick_Times,
                             cluster_Start_Time, cluster_End_Time,
                             waveforms, threshold, duplicate_Threshold,
                             DBSCAN_Distance, ax_DBSCAN0):
    """ Plots results of associate_Picks function. Displays and saves the
        figure on the last iteration. """

    # rows=1, columns are different time windows
    figureWidth = 49 # was 19
    figureHeight = 14

    # initialize figure on first period (or column)
    if period_Index == 0:
        fig, ax = plt.subplots(1, figure_Columns, sharex='col',
                               sharey='row', figsize=(figureWidth, figureHeight))

    # build list of subplot indices to build figure
    subplot_Indices = [index for index in range(1, figure_Columns + 1)]

    subplot_Index = subplot_Indices[period_Index]

    # set figure subplot, if first index set with different name to
    # enforce sharing of y axis
    if subplot_Index == subplot_Indices[0]:
        ax_DBSCAN0 = plt.subplot(1, figure_Columns,
                                 subplot_Index)
    else:
        ax_DBSCAN = plt.subplot(1, figure_Columns,
                                subplot_Index, sharey=ax_DBSCAN0)

    # set colors: black is removed and used for noise
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    # convert to np.array for bool indexing
    pick_Station_Order = np.array(pick_Station_Order)
    pick_Times = np.array(pick_Times)

    # calculate median time of picks for each cluster
    cluster_Medians = {}
    cluster_Means = {}
    for label in unique_labels:
        member_Indices = np.where(labels == label)[0]
        cluster_Medians[label] = np.median(pick_Times[member_Indices])
        cluster_Means[label] = np.mean(pick_Times[member_Indices])

    # loop through all clusters and plot them
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        # mask to identify members of class
        class_member_mask = (labels == k)

        # get cluster median and mean
        cluster_Median = cluster_Medians[k]
        cluster_Mean = cluster_Means[k]

        # # plot in PCA space
        # xy = pick_Array_PCA[class_member_mask & core_samples_mask]
        # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        #          markeredgecolor='k', markersize=9)
        #
        # xy = pick_Array_PCA[class_member_mask & ~core_samples_mask]
        # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        #          markeredgecolor='k', markersize=6)

        if k == -1:
            # plot noise
            y = pick_Station_Order[class_member_mask]  # & ~core_samples_mask
            x = pick_Times[class_member_mask]  # & ~core_samples_mask
            plt.plot_date(x, y, 'o', markerfacecolor='grey',
                          markeredgecolor='grey', markersize=6)

        else:
            # plot in station order vs. time space
            y = pick_Station_Order[class_member_mask]  # & core_samples_mask
            x = pick_Times[class_member_mask]  # & core_samples_mask
            plt.plot_date(x, y, 'o', markerfacecolor=tuple(col),
                          markeredgecolor='k', markersize=9)
            plt.plot_date(x, y, '|', markerfacecolor='k',
                          markeredgewidth=0.2, markeredgecolor='k',
                          markersize=9)

            # plot cluster median
            y2 = [pick_Station_Order[class_member_mask].min(),
                  pick_Station_Order[class_member_mask].max()]
            x2 = [cluster_Median, cluster_Median]
            plt.plot_date(x2, y2, fmt='-', linewidth=0.4, color='grey')

            # plot cluster mean
            x3 = [cluster_Mean, cluster_Mean]
            plt.plot_date(x3, y2, fmt='-', linewidth=0.4, color='green')

        # set axes attributes on first subplot
        if subplot_Index == subplot_Indices[0]:
            ax_DBSCAN0.set_yticks(np.arange(0, len(station_Distance)))
            ax_DBSCAN0.set_yticklabels(station_Distance.keys())
            ax_DBSCAN0.set_ylabel('Station')
            # ax_DBSCAN0.set_xticks(np.arange(0, 1.1, 0.5))
            ax_DBSCAN0.set_xlim([cluster_Start_Time.matplotlib_date,
                                 cluster_End_Time.matplotlib_date])
            # ax_DBSCAN0.set_xticklabels(np.arange(0, 1.1, 0.2))
            # ax_DBSCAN0.set_xlabel('Seconds')

            myFmt = DateFormatter("%S.")  # "%H:%M:%S.f"
            ax_DBSCAN0.xaxis.set_major_formatter(myFmt)
            locator = AutoDateLocator(minticks=5, maxticks=6)
            ax_DBSCAN0.xaxis.set_major_locator(locator)
            ax_DBSCAN0.set_ylim((-1, len(station_Distance)))
            # ax_DBSCAN0.set_xlim((0, 1))
            # fig.tight_layout()

            # reverse y axis
            ax_DBSCAN0.set_ylim(ax_DBSCAN0.get_ylim()[::-1])

        else:
            ax_DBSCAN.set_xlim([cluster_Start_Time.matplotlib_date,
                                cluster_End_Time.matplotlib_date])
            myFmt = DateFormatter("%S.")  # "%H:%M:%S.f"
            ax_DBSCAN.xaxis.set_major_formatter(myFmt)
            locator = AutoDateLocator(minticks=5, maxticks=6)
            ax_DBSCAN.xaxis.set_major_locator(locator)
            # ax_DBSCAN.set_xticklabels(np.arange(0, 1.1, 0.2))
            # ax_DBSCAN.set_xlabel('Seconds')
            ax_DBSCAN.yaxis.set_visible(False)

    # plot waveforms for each station
    # normalize by max value in each window
    maxTraceValue = max_Amplitude(waveforms)
    for trace in range(len(waveforms)):
        plt.plot_date(waveforms[trace].times(type="matplotlib"),
                      (waveforms[trace].data / maxTraceValue) * 1.25 +
                      trace, fmt="k-", linewidth=0.4)

    print(f"Column: {period_Index}")
    print(f"Start time: {cluster_Start_Time}")
    print(f"End time: {cluster_End_Time}")

    # display and save plot on last period_Index (last column)
    if subplot_Index == subplot_Indices[-1]:
        plt.xlabel('Time (seconds)')
        plt.suptitle('Association results', fontweight="bold", y=0.9999)
        plt.tight_layout()
        # remove gap between subplots
        plt.subplots_adjust(wspace=0.00)
        plt.savefig(f'clustering_Results_thr{threshold}_dupThr'
                    f'{duplicate_Threshold}_DB{DBSCAN_Distance}.pdf', dpi=600)
        plt.show()

    # return shared axis
    return ax_DBSCAN0

# function to ingest snuffler-type mrkr file
def process_picks(marker_File_Path: str):
    """take in a snuffler marker file and returns the count and dates of
    detected signals per station, and the dates of detected events
    """
    # build dict of [pick counts per station, [pick dates per station]]
    pick_counts_dates = {'1': [0, []], '2': [0, []], '3': [0, []],
                         '4': [0, []], '5': [0, []], '6': [0, []],
                         '7': [0, []], '8': [0, []], '9': [0, []],
                         '10': [0, []], '12': [0, []], '13': [0, []],
                         '14': [0, []],
                         '15': [0, []], '16': [0, []], '17': [0, []],
                         '18': [0, []], '20': [0, []], '21': [0, []],
                         '22': [0, []], '23': [0, []], '24': [0, []],
                         '25': [0, []],
                         '26': [0, []], '27': [0, []], '28': [0, []],
                         '29': [0, []],
                         '30': [0, []], '31': [0, []], '32': [0, []],
                         '33': [0, []], '34': [0, []], '35': [0, []],
                         '36': [0, []], '37': [0, []], '38': [0, []],
                         '39': [0, []], '40': [0, []], '41': [0, []],
                         '42': [0, []], 'UGAP3': [0, []], 'UGAP5': [0, []],
                         'UGAP6': [0, []]}
    event_dates = []

    # read the marker file line by line
    with open(marker_File_Path, 'r') as file:
        for line_Contents in file:
            if len(line_Contents) > 52: # avoid irrelevant short lines
                if (line_Contents[0:5] == 'phase') and (line_Contents[-20:-19] == 'P'): # if the line is a P wave pick
                    if len(line_Contents[36:52].split('.')) > 1: # avoids
                        # error from "None" contents
                        pick_station = line_Contents[36:52].split('.')[1]
                        # pick_Channel = line_Contents[36:52].split('.')[3]
                        # pick_Channel = pick_Channel.split(' ')[0]
                        pick_time = UTCDateTime(line_Contents[7:32])

                        # increment pick counter
                        pick_counts_dates[pick_station][0] += 1
                        # add pick time to list
                        pick_counts_dates[pick_station][1].append(pick_time)

                elif line_Contents[0:5] == 'event':
                    # extract event time and append to list
                    event_time = UTCDateTime(line_Contents[7:32])
                    event_dates.append(event_time)

    return pick_counts_dates, event_dates

# plot event histogram from mrkr file
def event_histogram(event_filename, save_fig=False):
    """Ingests snuffler-style .mrkr file and generates a histogram from its
       contents

       Example:
            event_filename = "/Users/human/Dropbox/Programs/unet/autopicked_events_03_13-07_09_2018.mrkr"
            event_histogram(event_filename, save_fig=True)
    """
    # get info from marker file
    _, event_dates = process_picks(event_filename)

    # convert event dates to matplotlib dates
    event_dates = [event_date.matplotlib_date for event_date in event_dates]

    # generate event date histogram
    # plt.figure(figsize=(9, 5)) # figsize for 1-2 weeks
    plt.figure(figsize=(13, 5))  # figsize for many weeks
    # 168 bins for every hour over 1 week
    # 336 bins for every 30 minutes over 1 week
    # 672 bins for every 15 minutes over 1 week
    # 1344 bins for every 15 minutes over 2 weeks
    # 2016 bins for every 15 minutes over 3 weeks
    # 2688 bins for every 15 minutes over 4 weeks
    # 3360 bins for every 15 minutes over 5 weeks
    # 4032 bins for every 15 minutes over 6 weeks
    # 4704 bins for every 15 minutes over 7 weeks
    # 5376 bins for every 15 minutes over 8 weeks
    # 6048 bins for every 15 minutes over 9 weeks
    # 6720 bins for every 15 minutes over 10 weeks
    # 7392 bins for every 15 minutes over 11 weeks
    # 8064 bins for every 15 minutes over 12 weeks
    # 2016 bins for every 1 hour over 12 weeks
    # 2184 bins for every 1 hour over 13 weeks
    # 2352 bins for every 1 hour over 14 weeks
    # 2520 bins for every 1 hour over 15 weeks
    # 2688 bins for every 1 hour over 16 weeks
    n, bins, patches = plt.hist(event_dates, bins=2688, facecolor="darkred",
                                alpha=0.6)
    ax = plt.axes()
    # set background color
    ax.set_facecolor("dimgrey")
    # set plot labels
    # plt.xlabel(f'hour : minute : second of'
    #            f' {num2date(bins[0]).strftime("%m/%d/%Y")}')
    plt.xlabel(f'Day [1 hour bins : 16 weeks starting on'
               f' {num2date(bins[0]).strftime("%m/%d/%Y")}]')
    plt.ylabel('Events')
    ax.set_title(f'Autodetected Events Histogram (n={len(event_dates)})',
                 y=0.9999)
    # set plot limits
    # plt.ylim(0, 50)
    ax.set_xlim([bins[0], bins[-1]])
    myFmt = DateFormatter("%m-%d")
    # myFmt = DateFormatter("%H:%M:%S")  # "%H:%M:%S.f"
    ax.xaxis.set_major_formatter(myFmt)
    locator_x = AutoDateLocator() # minticks=12, maxticks=18
    ax.xaxis.set_major_locator(locator_x)

    plt.tight_layout()
    plt.grid(True)

    if save_fig:
        plt.savefig(f'event_histogram.pdf', dpi=600)
    plt.show()

    return None

# plot signal histogram from mrkr file
def signal_histogram(event_filename, save_fig=False):
    """Ingests snuffler-style .mrkr file and generates a histogram from its
       contents

       Example:
            event_filename = "/Users/human/Dropbox/Programs/unet/autopicked_events_03_13-07_09_2018.mrkr"
            signal_histogram(event_filename, save_fig=True)
    """
    # get info from marker file
    pick_counts_dates, _ = process_picks(event_filename)

    event_dates = []
    # get dates of signals from dict
    for station in pick_counts_dates.keys():
        for signal_time in pick_counts_dates[station][1]:
            event_dates.append(signal_time)

    # convert event dates to matplotlib dates
    event_dates = [event_date.matplotlib_date for event_date in event_dates]

    # generate event date histogram
    # plt.figure(figsize=(9, 5)) # figsize for 1-2 weeks
    plt.figure(figsize=(13, 5))  # figsize for many weeks
    # 168 bins for every hour over 1 week
    # 336 bins for every 30 minutes over 1 week
    # 672 bins for every 15 minutes over 1 week
    # 1344 bins for every 15 minutes over 2 weeks
    # 2016 bins for every 15 minutes over 3 weeks
    # 2688 bins for every 15 minutes over 4 weeks
    # 3360 bins for every 15 minutes over 5 weeks
    # 4032 bins for every 15 minutes over 6 weeks
    # 4704 bins for every 15 minutes over 7 week
    # 5376 bins for every 15 minutes over 8 weeks
    # 6048 bins for every 15 minutes over 9 weeks
    # 6720 bins for every 15 minutes over 10 weeks
    # 7392 bins for every 15 minutes over 11 weeks
    # 8064 bins for every 15 minutes over 12 weeks
    # 2016 bins for every 1 hour over 12 weeks
    # 2184 bins for every 1 hour over 13 weeks
    # 2352 bins for every 1 hour over 14 weeks
    # 2520 bins for every 1 hour over 15 weeks
    # 2688 bins for every 1 hour over 16 weeks
    n, bins, patches = plt.hist(event_dates, bins=2688, facecolor="darkred",
                                alpha=0.6)
    ax = plt.axes()
    # set background color
    ax.set_facecolor("dimgrey")
    # set plot labels
    # plt.xlabel(f'hour : minute : second of'
    #            f' {num2date(bins[0]).strftime("%m/%d/%Y")}')
    plt.xlabel(f'Day [1 hour bins : 16 weeks starting on'
               f' {num2date(bins[0]).strftime("%m/%d/%Y")}]')
    plt.ylabel("P wave arrivals")
    ax.set_title(f'Autodetected Signals Histogram (n={len(event_dates)})',
                 y=0.9999)
    # set plot limits
    # plt.ylim(0, 50)
    ax.set_xlim([bins[0], bins[-1]])
    myFmt = DateFormatter("%m-%d")
    # myFmt = DateFormatter("%H:%M:%S")  # "%H:%M:%S.f"
    ax.xaxis.set_major_formatter(myFmt)
    locator_x = AutoDateLocator() # minticks=12, maxticks=18
    ax.xaxis.set_major_locator(locator_x)

    plt.tight_layout()
    plt.grid(True)

    if save_fig:
        plt.savefig(f'signal_histogram.pdf', dpi=600)
    plt.show()

    return None


if RUN:
    # # # #   B E G I N   P R O C E S S I N G   T I M E   W I N D O W S   # # # #
    start = time.time()

    # identify time period(s) of interest from input
    # 2018-05-08T00:01:10.0Z - 2018-05-08T12:00:00.0Z
    # 2018-05-08T12:00:00.0Z - 2018-05-08T23:59:00.0Z
    start_date = UTCDateTime(sys.argv[1])
    end_date = UTCDateTime(sys.argv[2])
    time_Period = (start_date, end_date)

    # build list of clustering algorithms to run
    clustering_Algos = []
    if _DBSCAN: clustering_Algos.append("DBSCAN")
    if _OPTICS: clustering_Algos.append("OPTICS")
    # FIXME: add spectral clustering & k-plane

    ###############################################################################
    # if clustering algos are specified, run them and generate figures to inspect #
    ###############################################################################
    # build tensorflow unet model
    model = build_unet_model()

    for clustering_Algo in clustering_Algos:

        # split into 5 windows for clustering
        clustering_Windows = int((time_Period[1] - time_Period[0]) / 5)

        #######################################################################
        ############# Loop through each time period and associate #############
        #######################################################################

        for period_Index in range(clustering_Windows):

            # time period for clustering & picking (5 second windows)
            cluster_Start_Time = time_Period[0] + period_Index * 5
            cluster_End_Time = cluster_Start_Time + 5

            # get project filepaths for the specified time period
            filepaths = project_Filepaths("Rattlesnake Ridge",
                                          cluster_Start_Time, cluster_End_Time)

            # load or create picking windows & trace data?
            if not load:
                # to save picking windows as files specify save_file=True in func
                picking_Windows, picking_Windows_Untransformed,\
                    picking_Windows_Times = find_Picking_Windows(filepaths,
                                                          detection_Parameters,
                                      [(cluster_Start_Time, cluster_End_Time)])

                # get stream of data for plotting
                channels = ['DP1', 'EHN']
                if plot_Association_Results:
                    waveforms = trim_Daily_Waveforms("Rattlesnake Ridge",
                                              cluster_Start_Time, cluster_End_Time,
                                                channels, write_File=False)

            else:
                # load picking windows
                infile = open(f'windows/picking_windows_'
                              f'{cluster_Start_Time}.pkl', 'rb')
                picking_Windows = pickle.load(infile)
                infile.close()

                # load raw data for windows
                infile = open(f'windows/picking_windows_untransformed'
                              f'_{cluster_Start_Time}.pkl','rb')
                picking_Windows_Untransformed = pickle.load(infile)
                infile.close()

                # load info associated with windows (times, station, channel)
                infile = open(f'windows/picking_times_'
                              f'{cluster_Start_Time}.pkl', 'rb')
                picking_Windows_Times = pickle.load(infile)
                infile.close()

                # format waveform filename
                waveformsFilename = str(cluster_Start_Time)[:19].replace(":",
                                                                       ".")
                # load 1s stream of data for plotting
                waveforms = read(f"waveforms/{waveformsFilename}.ms")

            ###################################################################
            #################### Get unet pick predictions ####################
            ###################################################################

            # get phase arrival time predictions from pretrained unet model
            pick_Predictions = get_Unet_Picks(picking_Windows,
                                              preloaded_model=model)

            # get only picks above specified threshold (determined from F1)
            picked_Windows, picked_Windows_Times, index_Of_Max_Prediction = \
                get_Threshold_Picks(pick_Predictions, threshold,
                                    picking_Windows_Untransformed,
                                    picking_Windows_Times)

            # find and remove duplicate predictions
            station_Picks, picked_Windows, picked_Windows_Times, \
            index_Of_Max_Prediction = remove_Duplicate_Picks(picked_Windows,
                                                          picked_Windows_Times,
                                                       index_Of_Max_Prediction)

            # print(f"Threshold: {threshold}\nPicks made: {len(picked_Windows)}")

            ###################################################################
            ####################### Visualize predictions #####################
            ###################################################################

            # plot 'em utilizing slicing (breaks when plotting > 172 lines)
            # plot_Picked_Windows(picked_Windows[0:50], index_Of_Max_Prediction[0:50])

            ###################################################################
            ############ Associate events with clustering & rules #############
            ###################################################################

            station_Distance, reverse_station_distance, pick_Station_Order, \
            pick_Times, labels = associate_Picks(clustering_Algo, station_Picks,
                                                 scale_Time, PCA, PCA_Components,
                                                 duplicate_Threshold,
                                                 temporal_Threshold,
                                                 spatial_Threshold,
                                                 min_Quakes_Per_Cluster, model)

            ###################################################################
            ################# Save the association results ####################
            ###################################################################

            # only save if there are picks in this window
            if len(labels) > 0:
                # define the filename and save the results
                event_filename = f"{time_Period[0].month:02}{time_Period[0].day:02}_{time_Period[0].hour:02}_{time_Period[1].hour:02}.mrkr"
                save_associations(event_filename, labels, pick_Times,
                                  pick_Station_Order, reverse_station_distance)

            ###################################################################
            ################# Plot the association results ####################
            ###################################################################

            if plot_Association_Results:
                # initialize axis object for first column
                if period_Index == 0:
                    ax_DBSCAN0 = []
                # build the figure for each window and display the figure on the
                # last iteration
                ax_DBSCAN0 = build_association_figure(clustering_Windows,
                                                      period_Index, labels,
                                                      pick_Station_Order, pick_Times,
                                                      cluster_Start_Time,
                                                      cluster_End_Time, waveforms,
                                                      threshold, duplicate_Threshold,
                                                      DBSCAN_Distance, ax_DBSCAN0)

        # if _OPTICS:
        #     # FIXME: this section needs updated

    end = time.time()
    hours = int((end - start) / 60 / 60)
    minutes = int(((end - start) / 60) - (hours * 60))
    seconds = int((end - start) - (minutes * 60) - (hours * 60 * 60))
    print(f"Runtime: {hours} h {minutes} m {seconds} s")
