"""
Functions to fetch data. 
"""

import obspy
from obspy.core.utcdatetime import UTCDateTime
from obspy.clients.fdsn import Client
import calendar
from datetime import datetime
import glob
import numpy as np
import utm
from obspy import read, Stream, Inventory
from obspy.core.utcdatetime import UTCDateTime
import os
from lidar import grids_from_raster
import netCDF4 as nc
import geopy
import proplot as pplt
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
from scipy import stats, interpolate
from scipy.signal import hilbert
from time_miner import build_unet_model, trace_arrival_prediction
import base64
import hashlib
from tqdm import tqdm
from eqcorrscan.utils.mag_calc import svd_moments
from eqcorrscan.utils.clustering import svd
from eqcorrscan import tests
import pandas as pd
import tensorflow as tf
# turn off verbose tensorflow logging
tf.get_logger().setLevel('INFO')

# helper function to get signal to noise ratio of time series
def snr(obspyObject: Stream or Trace) -> float:
    '''
    (obspyObject) -> float

    Returns the signal to noise ratios of each trace of a specified obspy
    stream object, or the specified obspy trace object. Signal-to-noise ratio
    defined as the ratio of the maximum amplitude in the timeseries to the rms
    amplitude in the entire timeseries.
    '''
    trace_rms = {}
    trace_maxAmplitude = {}

    # for Stream objects
    if isinstance(obspyObject, obspy.core.stream.Stream):
        for index, trace in enumerate(obspyObject):
            rms = np.sqrt(np.mean(trace.data ** 2))
            maxAmplitude = abs(trace.max())
            trace_rms.update({index: rms})
            trace_maxAmplitude.update({index: maxAmplitude})

        snrs = [trace_maxAmplitude[key] / trace_rms[key] for key in
                trace_rms]

    # for Trace objects
    elif isinstance(obspyObject, obspy.core.trace.Trace):
        if len(obspyObject.data) > 0:
            rms = np.sqrt(np.mean(obspyObject.data ** 2))
            maxAmplitude = abs(obspyObject.max())
            trace_rms.update({obspyObject.id: rms})
            trace_maxAmplitude.update({obspyObject.id: maxAmplitude})
            snrs = [trace_maxAmplitude[key] / trace_rms[key] for key in
                    trace_rms]
        else:
            snrs = []

    return snrs


# helper function to find max amplitude of a time series for plotting
def max_amplitude(timeSeries):
    '''
    (np.ndarray or obspy stream or trace object) -> (float)

    Determines single max value that occurs in a numpy array or trace or in
    all traces of a stream. Returns the max value and its index.
    '''
    # for Stream objects
    if isinstance(timeSeries, obspy.core.stream.Stream):
        # creates a list of the max value in each trace
        traceMax = [np.nanmax(np.abs(timeSeries[trace].data)) for trace in
                    range(len(timeSeries))]
        # return max value among all traces
        return np.max(traceMax), np.nanargmax(traceMax)

    # for Trace objects
    elif isinstance(timeSeries, obspy.core.trace.Trace):
        if len(timeSeries.data) > 0:
            # guard against all NaN arrays
            if not np.all(np.isnan(timeSeries.data)):
                try:
                    traceMax = np.nanmax(np.abs(timeSeries.data))
                    max_index = np.nanargmax(np.abs(timeSeries.data))
                except Exception:
                    traceMax = None
                    max_index = None
                    pass
            else:
                traceMax = None
                max_index = None
        else:
            traceMax = None
            max_index = None

        return traceMax, max_index # return max value and index

    # FIXME add max index to return for ndarray case
    elif isinstance(timeSeries, np.ndarray):
        # returns the max for each row, works with 1D and 2D np.ndarrays
        if len(timeSeries.shape) == 1:  # 1D case
            return np.abs(timeSeries).max()
        elif len(timeSeries.shape) == 2:  # 2D case
            return np.abs(timeSeries).max(1)
        else:
            print("You broke the max_amplitude function with a np.ndarray "
                  "that is not 1D or 2D ;(")

# downloads time series data from IRIS DMC
def get_Waveforms(network, stations, location, channels, start_Time,
                  end_Time, format=None):
    """Downloads waveform data using Obspy to access the IRIS DMC and saves
    data to daily miniseed files.
    Find stations via https://ds.iris.edu/gmap/

    Example: download UGAP3, UGAP5, UGAP6  :  EHN [old DP1], EHE, EHZ
        network = "UW"
        stations = ["UGAP3", "UGAP5", "UGAP6"]
        location = "**"
        channels = ["EHN", "EHE", "EHZ"]
        start_Time = UTCDateTime("2018-05-08T00:00:00.0Z")
        end_Time =   UTCDateTime("2018-05-08T23:59:59.999999999999999Z") # 6/4
        get_Waveforms(network, stations, location, channels, start_Time, end_Time)

    Example: download UGAP3, UGAP5, UGAP6  :  EHN, format for RR talapas runs
            network = "UW"
            stations = ["UGAP5"]# ["UGAP3", "UGAP5", "UGAP6"]
            location = "**"
            channels = ["EHN"]
            start_Time = UTCDateTime("2018-05-09T00:00:00.0Z")
            end_Time =   UTCDateTime("2018-05-14T23:59:59.999999999999999Z")
            get_Waveforms(network, stations, location, channels, start_Time,
                          end_Time, format="RR")

    Example: download stations near Wrangell Volcanic Field inside of box:
              North: 63.9498, East: -141.5259, South: 61.0264, West: -149.3515

        # all this data must be downloaded in two parts due to Obspy
        # limitations on miniseed file size
        network = "AK"
        stations = ["BAL", "BARN", "DHY", "DIV", "DOT", "GHO", "GLB", "K218",
                    "KLU", "KNK", "MCAR", "MCK", "PAX", "PTPK", "RIDG", "RND",
                    "SAW", "SCM", "VRDI", "WAT1", "WAT6", "WAT7"]
        location = "**"
        channels = ["BHZ", "BNZ", "HNZ", "BHN", "BNN", "HNN", "BHE", "BNE", "HNE"]
        start_Time = UTCDateTime("2016-06-15T00:00:00.0Z")
        end_Time =   UTCDateTime("2017-08-11T23:59:59.999999999999999Z")
        get_Waveforms(network, stations, location, channels, start_Time, end_Time)
        start_Time = UTCDateTime("2017-08-12T00:00:00.0Z")
        end_Time =   UTCDateTime("2018-08-11T23:59:59.999999999999999Z")
        get_Waveforms(network, stations, location, channels, start_Time, end_Time)

        # and subsequent networks
        network = "AT"
        stations = ["MENT", "PMR"]
        location = "**"
        channels = ["BHZ", "BHN", "BHE"]
        start_Time = UTCDateTime("2016-06-15T00:00:00.0Z")
        end_Time =   UTCDateTime("2017-08-11T23:59:59.999999999999999Z")
        get_Waveforms(network, stations, location, channels, start_Time, end_Time)
        start_Time = UTCDateTime("2017-08-12T00:00:00.0Z")
        end_Time =   UTCDateTime("2018-08-11T23:59:59.999999999999999Z")
        get_Waveforms(network, stations, location, channels, start_Time, end_Time)

        network = "AV"
        stations = ["WACK", "WASW", "WAZA"]
        location = "**"
        channels = ["BHZ", "SHZ", "BHN", "SHN", "BHE", "SHE"]
        start_Time = UTCDateTime("2016-07-17T00:00:00.0Z")
        end_Time =   UTCDateTime("2017-08-11T23:59:59.999999999999999Z")
        get_Waveforms(network, stations, location, channels, start_Time, end_Time)
        start_Time = UTCDateTime("2017-08-12T00:00:00.0Z")
        end_Time =   UTCDateTime("2018-08-11T23:59:59.999999999999999Z")
        get_Waveforms(network, stations, location, channels, start_Time, end_Time)

        network = "NP"
        stations = ["2730", "2738", "2784", "8034", "AJKS", "AMJG"]
        location = "**"
        channels = ["HNZ", "HNN", "HNE"]
        start_Time = UTCDateTime("2016-06-15T00:00:00.0Z")
        end_Time =   UTCDateTime("2017-08-11T23:59:59.999999999999999Z")
        get_Waveforms(network, stations, location, channels, start_Time, end_Time)
        start_Time = UTCDateTime("2017-08-12T00:00:00.0Z")
        end_Time =   UTCDateTime("2018-08-11T23:59:59.999999999999999Z")
        get_Waveforms(network, stations, location, channels, start_Time, end_Time)

        network = "TA"
        stations = ["HARP", "K24K", "L26K", "L27K", "M23K", "M24K", "M26K",
                    "M27K", "N25K"]
        location = "**"
        channels = ["BHZ", "BHN", "BHE"]
        start_Time = UTCDateTime("2016-06-15T00:00:00.0Z")
        end_Time =   UTCDateTime("2017-08-11T23:59:59.999999999999999Z")
        get_Waveforms(network, stations, location, channels, start_Time, end_Time)
        start_Time = UTCDateTime("2017-08-12T00:00:00.0Z")
        end_Time =   UTCDateTime("2018-08-11T23:59:59.999999999999999Z")
        get_Waveforms(network, stations, location, channels, start_Time, end_Time)

        # WVLF experiment
        network = "YG"
        stations = ["DEN1", "DEN3", "DEN4", "DEN5", "GLN1", "GLN2", "GLN3", "GLN4",
                    "LKLO", "MCR1", "MCR2", "MCR3", "MCR4", "NEB1", "NEB2", "NEB3",
                    "RH01", "RH02", "RH03", "RH04", "RH05", "RH06", "RH07", "RH08",
                    "RH09", "RH10", "RH11", "RH12", "RH13", "RH14", "RH15", "TOK1",
                    "TOK2", "TOK3", "TOK4", "TOK5"]
        location = "**"
        channels = ["BHZ", "BHN", "BHE"]
        start_Time = UTCDateTime("2016-06-15T00:00:00.0Z")
        end_Time =   UTCDateTime("2017-08-11T23:59:59.999999999999999Z")
        get_Waveforms(network, stations, location, channels, start_Time, end_Time)
        start_Time = UTCDateTime("2017-08-12T00:00:00.0Z")
        end_Time =   UTCDateTime("2018-08-11T23:59:59.999999999999999Z")
        get_Waveforms(network, stations, location, channels, start_Time, end_Time)

    # Another example: download data for a single time period

        network = "AK"
        stations = ["BAL", "BARN", "DHY", "DIV", "DOT", "GHO", "GLB", "K218",
                    "KLU", "KNK", "MCAR", "MCK", "PAX", "PTPK", "RIDG", "RND",
                    "SAW", "SCM", "VRDI", "WAT1", "WAT6", "WAT7"]
        location = "**"
        channels = ["BHZ", "BNZ", "HNZ", "BHN", "BNN", "HNN", "BHE", "BNE", "HNE"]
        start_Time = UTCDateTime("2016-09-26T00:00:00.0Z")
        end_Time =   UTCDateTime("2016-09-30T23:59:59.999999999999999Z")
        get_Waveforms(network, stations, location, channels, start_Time, end_Time)

        # and subsequent networks
        network = "AT"
        stations = ["MENT", "PMR"]
        location = "**"
        channels = ["BHZ", "BHN", "BHE"]
        get_Waveforms(network, stations, location, channels, start_Time, end_Time)

        network = "AV"
        stations = ["WACK", "WASW", "WAZA"]
        location = "**"
        channels = ["BHZ", "SHZ", "BHN", "SHN", "BHE", "SHE"]
        get_Waveforms(network, stations, location, channels, start_Time, end_Time)

        network = "NP"
        stations = ["2730", "2738", "2784", "8034", "AJKS", "AMJG"]
        location = "**"
        channels = ["HNZ", "HNN", "HNE"]
        get_Waveforms(network, stations, location, channels, start_Time, end_Time)

        network = "TA"
        stations = ["HARP", "K24K", "L26K", "L27K", "M23K", "M24K", "M26K",
                    "M27K", "N25K"]
        location = "**"
        channels = ["BHZ", "BHN", "BHE"]
        get_Waveforms(network, stations, location, channels, start_Time, end_Time)

        # WVLF experiment
        network = "YG"
        stations = ["DEN1", "DEN3", "DEN4", "DEN5", "GLN1", "GLN2", "GLN3", "GLN4",
                    "LKLO", "MCR1", "MCR2", "MCR3", "MCR4", "NEB1", "NEB2", "NEB3",
                    "RH01", "RH02", "RH03", "RH04", "RH05", "RH06", "RH07", "RH08",
                    "RH09", "RH10", "RH11", "RH12", "RH13", "RH14", "RH15", "TOK1",
                    "TOK2", "TOK3", "TOK4", "TOK5"]
        location = "**"
        channels = ["BHZ", "BHN", "BHE"]
        get_Waveforms(network, stations, location, channels, start_Time, end_Time)

    """

    if isinstance(stations, list):
        for station in stations:
            for channel in channels:
                client = Client("IRIS")
                try:
                    st = client.get_waveforms(network, station, location, channel,
                                              start_Time, end_Time)
                    # fix rounding errors in sampling rate before merging
                    for index, trace in enumerate(st):
                        st[index].stats.sampling_rate = round(
                            trace.stats.sampling_rate)

                    st = st.merge(method=1, fill_value=0) # None
                    # st.plot(type='dayplot', interval=60)
                    # print(f"len of st: {len(st)}")
                    print(st)

                    stream_start = st[0].stats.starttime
                    stream_end = st[0].stats.endtime

                    # loop over stream time period and write daily files
                    # first initialize file start date
                    file_start = UTCDateTime(f"{stream_start.year}-"
                                             f"{stream_start.month:02}-"
                                             f"{stream_start.day:02}"
                                             f"T00:00:00.0Z")

                    # keep looping until last day is reached
                    while file_start < stream_end:
                        if format == "RR":
                            filename = f"5A.{station}..{channel}" \
                                       f".{file_start.year}" \
                                       f"-{file_start.month:02}-" \
                                       f"{file_start.day:02}T00.00.00.ms"
                        else:
                            filename = f"{network}.{station}.{channel}" \
                                       f".{file_start.year}" \
                                       f"-{file_start.month:02}-" \
                                       f"{file_start.day:02}.ms"

                        file_st = st.copy()
                        file_end = UTCDateTime(f"{file_start.year}-"
                                             f"{file_start.month:02}-"
                                             f"{file_start.day:02}"
                                             f"T23:59:59.999999999Z")
                        file_st.trim(file_start, file_end,
                                     nearest_sample=False, fill_value=None)

                        file_st.write(filename, format="MSEED")
                        print(f"Writing {filename}")

                        # update the next file start date
                        file_year = file_start.year
                        file_month = file_start.month
                        file_day = file_start.day
                        # get number of days in current month
                        month_end_day = calendar.monthrange(file_year,
                                                            file_month)[1]
                        # check if year should increment
                        if file_month == 12 and file_day == month_end_day:
                            file_year += 1
                            file_month = 1
                            file_day = 1
                        # check if month should increment
                        elif file_day == month_end_day:
                            file_month += 1
                            file_day = 1
                        # else only increment day
                        else:
                            file_day += 1

                        file_start = UTCDateTime(f"{file_year}-"
                                                 f"{file_month:02}-"
                                                 f"{file_day:02}T00:00:00.0Z")
                except Exception:
                    pass

    else: # better be a string of a single station or BOOM broken
        for channel in channels:
            client = Client("IRIS")
            try:
                st = client.get_waveforms(network, stations, location, channel,
                                          start_Time, end_Time)
                # fix rounding errors in sampling rate before merging
                for index, trace in enumerate(st):
                    st[index].stats.sampling_rate = round(
                        trace.stats.sampling_rate)

                st = st.merge(method=1, fill_value=None)
                # st.plot(type='dayplot', interval=60)

                stream_start = st[0].stats.starttime
                stream_end = st[0].stats.endtime

                # loop over stream time period and write daily files
                # first initialize file start date
                file_start = UTCDateTime(f"{stream_start.year}-"
                                         f"{stream_start.month:02}-"
                                         f"{stream_start.day:02}T00:00:00.0Z")

                # keep looping until last day is reached
                while file_start < stream_end:

                    filename = f"{network}.{stations}.{channel}" \
                               f".{file_start.year}" \
                               f"-{file_start.month:02}-" \
                               f"{file_start.day:02}.ms"

                    file_st = st.copy()
                    file_end = UTCDateTime(f"{file_start.year}-"
                                           f"{file_start.month:02}-"
                                           f"{file_start.day:02}"
                                           f"T23:59:59.999999999Z")
                    file_st.trim(file_start, file_end,
                                 nearest_sample=False, fill_value=None)

                    file_st.write(filename, format="MSEED")

                    # update the next file start date
                    file_year = file_start.year
                    file_month = file_start.month
                    file_day = file_start.day
                    # get number of days in current month
                    month_end_day = calendar.monthrange(file_year,
                                                        file_month)[1]
                    # check if year should increment
                    if file_month == 12 and file_day == month_end_day:
                        file_year += 1
                        file_month = 1
                        file_day = 1
                    # check if month should increment
                    elif file_day == month_end_day:
                        file_month += 1
                        file_day = 1
                    # else only increment day
                    else:
                        file_day += 1

                    file_start = UTCDateTime(f"{file_year}-"
                                             f"{file_month:02}-"
                                             f"{file_day:02}T00:00:00.0Z")
            except Exception:
                pass


def get_Events(startTime: datetime, endTime: datetime, minLongitude: float,
               maxLongitude: float, minLatitude: float, maxLatitude: float,
               maxDepth: float):

    # Use comcat to pull events from server
    def get_focal(row):
        # there may be more than one source column for focal angles
        # in a detail data frame, so we're filtering out the NaN
        # columns in this row
        strike_cols = row.filter(regex='np1_strike').values
        strike_cols = strike_cols.astype(np.float64)
        strike = strike_cols[~np.isnan(strike_cols)]
        if len(strike) == 0: # guards against return NaNs
            return None
        strike = strike_cols[~np.isnan(strike_cols)][0]

        dip_cols = row.filter(regex='np1_dip').values
        dip_cols = dip_cols.astype(np.float64)
        dip = dip_cols[~np.isnan(dip_cols)]
        if len(dip) == 0:
            return None
        dip = dip_cols[~np.isnan(dip_cols)][0]

        rake_cols = row.filter(regex='np1_rake').values
        rake_cols = rake_cols.astype(np.float64)
        rake = rake_cols[~np.isnan(rake_cols)]
        if len(rake) == 0:
            return None
        rake = rake_cols[~np.isnan(rake_cols)][0]

        return (strike, dip, rake)

    # find events of interest
    print(f"Searching for events.")
    events = search(starttime=startTime,
                    endtime=endTime,
                    minlatitude=minLatitude,
                    maxlatitude=maxLatitude,
                    minlongitude=minLongitude,
                    maxlongitude=maxLongitude,
                    maxdepth=maxDepth,
                    minmagnitude=3.0,
                    maxmagnitude=9.9,
                    producttype='moment-tensor')

    print(f"Found {len(events)} potential events.")
    print("Downloading event data.")
    detail_events = get_detail_data_frame(events)

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson()) # projection=ccrs.Robinson()
    # ax.set_global() # global map rather than data extent
    ax.stock_img()
    ax.coastlines()

    event_count = 0
    longitude_array = []
    latitude_array = []
    depth_array = []
    strike_array = []
    dip_array = []
    rake_array = []
    # FIXME make figure better using geospatial_Figure function below
    for idx, row in detail_events.iterrows():
        foc = get_focal(row)
        if foc is None: # skips lines with NaN values for strike, dip, and/or rake
            continue
        event_count += 1
        x, y = ax.projection.transform_point(row['longitude'],
                                             row['latitude'],
                                             ccrs.PlateCarree())
        # add x and y to lon and lat list (to later find min and max for extent)
        longitude_array.append(row['longitude'])
        latitude_array.append(row['latitude'])
        depth_array.append(row['depth'])
        strike_array.append(foc[0])
        dip_array.append(foc[1])
        rake_array.append(foc[2])

        b = beach(foc, xy=(x, y), width=60000, linewidth=1)
        b.set_zorder(100)
        ax.add_collection(b)

    print(f'Found {event_count} events with focal mechanisms.')
    longitude_array = np.array(longitude_array)
    latitude_array = np.array(latitude_array)
    # min lon, max lon, min lat, max lat : set extent of figure based on stored values
    buffer = 1
    extent = [np.amin(longitude_array) - buffer, np.amax(longitude_array) + buffer,
              np.amin(latitude_array) - buffer, np.amax(latitude_array) + buffer]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    plt.show()
    # plt.savefig('focal_mech_map.pdf')

     # return np.array with columns: longitude, latitude, depth, strike, dip, rake
    depth_array = np.array(depth_array)
    strike_array = np.array(strike_array)
    dip_array = np.array(dip_array)
    rake_array = np.array(rake_array)
    return np.transpose(np.vstack((longitude_array, latitude_array, depth_array, strike_array, dip_array, rake_array)))


def trim_Daily_Waveforms(project_Name: str, start_Time, end_Time, channels:
                         list, write_File=False):
    '''loads project data into an Obspy Stream object. By default this will grab the entire day.
    If start_Time and end_Time are specified the Stream object will be trimmed to span that period.
    start_Time and end_Time are UTCDateTime objects

    Example: for all stations in one stream with distance hack, for picking
    start_Time = UTCDateTime("2018-03-13T00:02:00.000Z")
    end_Time =   UTCDateTime("2018-03-13T00:05:00.000Z")
    project_Name = "Rattlesnake Ridge"
    channels = ['DP1', 'EHN']
    trim_Daily_Waveforms(project_Name, start_Time, end_Time, channels, write_File=True)

    start_Time = UTCDateTime("2018-04-20T18:10:00.0Z")
    end_Time =   UTCDateTime("2018-04-20T18:14:01.0Z")
    project_Name = "Rattlesnake Ridge"
    channels = ['DP1', 'EHN']
    trim_Daily_Waveforms(project_Name, start_Time, end_Time, 
    channels, write_File=True)

    start_Time = UTCDateTime("2018-04-20T11:10:00.0Z")
    end_Time =   UTCDateTime("2018-04-20T11:14:01.0Z")
    project_Name = "Rattlesnake Ridge"
    channels = ['DP1', 'EHN']
    trim_Daily_Waveforms(project_Name, start_Time, end_Time, 
    channels, write_File=True)

    Example:
    start_Time = UTCDateTime("2018-03-13T00:04:00.0Z")
    end_Time =   UTCDateTime("2018-03-13T00:06:00.0Z")
    project_Name = "Rattlesnake Ridge"
    channels = ['DP1', 'DP2', 'DPZ', 'EHN', 'EHE', 'EHZ']
    trim_Daily_Waveforms(project_Name, start_Time, end_Time, 
    channels, write_File=True)

    start_Time = UTCDateTime("2016-09-26T09:26:00.0Z")
    end_Time =   UTCDateTime("2016-09-26T09:31:00.0Z")
    project_Name = "Alaska"
    channels = []
    trim_Daily_Waveforms(project_Name, start_Time, end_Time, channels, write_File=True)

    #FIXME currently limited to same-day queries, use if endtime-starttime > 1 day:

    '''
    project_Aliases = {"Rattlesnake Ridge": "RR"}

    if project_Name == "Rattlesnake Ridge":
        # build filepath list based on dates, station type, and channels
        node = ['DP1', 'DP2', 'DPZ']  # nodal seismometer channels
        ugap = ['EHN', 'EHE', 'EHZ']
        # order matters for distance hack
        stations_channels = {'1': node, '2': node, '3': node, '5': node,
                             '4': node, '6': node, '7': node, '8': node,
                             '13': node, '9': node, '10': node, '12': node,
                             '15': node, 'UGAP3': ugap, '16': node, '17': node,
                             '18': node, '20': node, '21': node, '22': node,
                             '23': node, '25': node, '26': node, '27': node,
                             'UGAP5': ugap, 'UGAP6': ugap, '28': node,
                             '30': node, '31': node, '32': node, '33': node,
                             '34': node, '35': node, '36': node, '37': node,
                             '38': node, '39': node, '40': node, '41': node,
                             '42': node}

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
                    filepath = f"/Users/human/Desktop/RR_MSEED/5A.{station}.." \
                               f"{channel}.{start_Time.year}-" \
                               f"{start_Time.month:02}-{start_Time.day:02}T00.00.00.ms"
                    filepaths.append(filepath)

        obspyStream = Stream()
        for filepath_idx in range(len(filepaths)):
            obspyStream += read(filepaths[filepath_idx]).merge(method=1,
                                                               fill_value=None)
            # station-distance hack for picking, assign number to network
            hack = station_distance_hack[obspyStream[
                filepath_idx].stats.station]
            obspyStream[filepath_idx].stats.network = f'{hack:02}'

    elif project_Name == "Alaska":
        # build stream of all station files for templates
        files_path = "/Users/human/ak_data/inner"
        filepaths = []
        for channel in channels:
            filepaths += glob.glob(f"{files_path}/*.{channel}"
                                   f".{start_Time.year}-{start_Time.month:02}-"
                                   f"{start_Time.day:02}.ms")

        obspyStream = Stream()
        for filepath_idx in range(len(filepaths)):
            st = read(filepaths[filepath_idx]).merge(method=1, fill_value=None)
            st.trim(start_Time - 30, end_Time + 30, pad=True,
                        fill_value=np.nan, nearest_sample=True)
            st.detrend()
            # interpolate to 100 Hz
            st.interpolate(sampling_rate=100.0)
            # bandpass filter
            st.filter('bandpass', freqmin=1, freqmax=15)

            obspyStream += st

    # make sure all traces have the same sampling rate (and thus number of
    # samples and length) to avoid bugs in other programs, e.g. Snuffler
    # slows down with variable sampling rates
    interpolate = False
    for index, trace in enumerate(obspyStream):
        # the sampling rate of the first trace is assumed to be correct
        if trace.stats.sampling_rate != obspyStream[0].stats.sampling_rate:
            print(f"Trace {index} has a different sampling rate. ")
            print(f"Station {trace.stats.station}, Channel "
                  f"{trace.stats.channel}, Start: "
                  f"{trace.stats.starttime}, End: {trace.stats.endtime}")
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
        obspyStream.trim(start_Time, end_Time, pad=True, fill_value=np.nan,
                         nearest_sample=True)

    if write_File:
        # format filename and save Stream as miniseed file
        start_Time_Stamp = str(obspyStream[0].stats.starttime)[
                           11:19].replace(":", ".") # use [:19] for date and time
        end_Time_Stamp = str(obspyStream[0].stats.endtime)[11:19].replace(
                                                                      ":", ".")
        # writes to snuffler path
        obspyStream.write(f"{start_Time_Stamp}_{end_Time_Stamp}.ms", format="MSEED")

    return obspyStream


def process_gmap_file(filepath):
    """ Processes the specified gmap-station.txt file and returns a
    dictionary of station keys and entries specifying the location and
    operating period of the station. GMAP example:
    https://ds.iris.edu/gmap/#network=5A&maxlat=46.5301&maxlon=-120.4604&minlat=46.5217&minlon=-120.4706&drawingmode=box&planet=earth

    Returns:
        dict of dict of list, e.g.
        # station   end time                start time         latitude      longitude  elevation
        {'1': {'2018-04-08T21:19:05': ['2018-03-12T18:05:50', '46.528358', '-120.466791', '966.5'],
               '2018-05-08T21:02:48': ['2018-04-08T21:21:01', '46.528349', '-120.466789', '966.5'],
               '2018-06-07T21:36:43': ['2018-05-08T21:05:18', '46.528371', '-120.466777', '966.5'],
               '2018-07-09T22:04:22': ['2018-06-07T21:38:48', '46.528372', '-120.466783', '966.5']},
         '2': {'2018-04-08T22:36:40': ['2018-03-12T17:56:16', '46.528279', '-120.466254', '963.9'],
               '2018-05-08T20:12:38': ['2018-04-08T22:38:23', '46.528289', '-120.466247', '963.9'],
               '2018-07-09T22:51:40': ['2018-06-07T22:05:28', '46.528294', '-120.466243', '963.9']}
        }

    Example:
        filepath = "/Users/human/Dropbox/Programs/stingray/projects/rattlesnake_ridge/gmap-stations.txt"
        location_dict = process_gmap_file(filepath)
    """
    # initialize a dict to store station information
    info_dict = {}

    # open file and loop through each line
    with open(filepath, 'r') as file:
        for line_contents in file:
            if len(line_contents) > 90:  # avoid irrelevant short lines
                content_list = line_contents.split("|")
                station = content_list[1]

                # get list associated with station or set to empty dict
                station_dict = info_dict.setdefault(station, {})
                # add line contents to station dict
                latitude = content_list[2]
                longitude = content_list[3]
                elevation = content_list[4]
                start_date = content_list[6]
                end_date = content_list[7][:-1] # strip newline character
                station_dict.update({end_date: [start_date, latitude,
                                                longitude, elevation]})

                # update info_dict with modified station_dict
                info_dict.update(station=station_dict)

    return info_dict


# function to return project stations active on a given date
def project_stations(project_name: str, date: UTCDateTime):
    """ Returns a list of the stations active on the specified date for the
    specified project in the linear scarp reference frame from ~North to
    ~South.
    """
    if project_name == "Rattlesnake Ridge":
        # # build filepath list based on dates
        # date = datetime(date_time.year, date_time.month, date_time.day)
        # if date <= datetime(2018, 4, 8):
        #     stas = [1, 2, 3, 5, 4, 6, 7, 8, 13, 9, 10, 12, 15, 'UGAP3', 16, 17,
        #             18, 20, 21, 22, 23, 25, 26, 27, 'UGAP5', 'UGAP6', 28, 30,
        #             31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
        # elif date <= datetime(2018, 5, 7):
        #     stas = [1, 2, 3, 5, 4, 6, 7, 8, 13, 9, 10, 12, 14, 15, 'UGAP3', 16,
        #             17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 'UGAP5', 'UGAP6',
        #             28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
        # elif date == datetime(2018, 5, 8):
        #     stas = [1, 3, 5, 4, 6, 7, 8, 13, 9, 10, 12, 14, 15, 'UGAP3', 16,
        #             17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 'UGAP5', 'UGAP6',
        #             28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
        # elif date <= datetime(2018, 6, 6):
        #     stas = [1, 3, 5, 4, 6, 7, 8, 13, 9, 10, 12, 14, 15, 'UGAP3', 16,
        #             17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 'UGAP5', 'UGAP6',
        #             28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
        # elif date == datetime(2018, 6, 7):
        #     stas = [1, 3, 5, 4, 6, 7, 8, 13, 9, 10, 14, 15, 'UGAP3', 16, 18,
        #             20, 21, 22, 23, 24, 25, 26, 27, 'UGAP5', 'UGAP6', 28, 29,
        #             30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42]
        # else:
        #     stas = [1, 2, 3, 5, 4, 6, 7, 8, 13, 9, 10, 14, 15, 'UGAP3', 16, 18,
        #             20, 21, 22, 23, 24, 25, 26, 27, 'UGAP5', 'UGAP6', 28, 29,
        #             30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42]

        # get time dependent stations. note: this is not robust so doing
        # analyses around the edges of deployments is a bad idea
        zeroed_date = UTCDateTime(f"{date.year}-{date.month:02}-"
                                  f"{date.day:02}T00:00:00.0Z")
        if date <= UTCDateTime("2018-04-08T22:00:00.0Z"):
            stas = [1, 2, 3, 5, 4, 6, 7, 8, 13, 9, 10, 12, 15, 'UGAP3', 16, 17,
                    18, 20, 21, 22, 23, 25, 26, 27, 'UGAP5', 'UGAP6', 28, 30,
                    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
        elif date <= UTCDateTime("2018-05-07T23:59:59.9999999999Z"):
            stas = [1, 2, 3, 5, 4, 6, 7, 8, 13, 9, 10, 12, 14, 15, 'UGAP3', 16,
                    17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 'UGAP5', 'UGAP6',
                    28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
        elif zeroed_date == UTCDateTime("2018-05-08T00:00:00.0Z"):
            stas = [1, 3, 5, 4, 6, 7, 8, 13, 9, 10, 12, 14, 15, 'UGAP3', 16,
                    17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 'UGAP5', 'UGAP6',
                    28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
        elif date <= UTCDateTime("2018-06-06T23:59:59.9999999999Z"):
            stas = [1, 3, 5, 4, 6, 7, 8, 13, 9, 10, 12, 14, 15, 'UGAP3', 16,
                    17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 'UGAP5', 'UGAP6',
                    28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
        elif zeroed_date == UTCDateTime("2018-06-07T00:00:00.0Z"):
            stas = [1, 3, 5, 4, 6, 7, 8, 13, 9, 10, 14, 15, 'UGAP3', 16, 18,
                    20, 21, 22, 23, 24, 25, 26, 27, 'UGAP5', 'UGAP6', 28, 29,
                    30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42]
        else:
            stas = [1, 2, 3, 5, 4, 6, 7, 8, 13, 9, 10, 14, 15, 'UGAP3', 16, 18,
                    20, 21, 22, 23, 24, 25, 26, 27, 'UGAP5', 'UGAP6', 28, 29,
                    30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42]

    return stas


def rattlesnake_Ridge_Station_Locations(date, format=None):
    """ Returns a dict of station locations, used by EQTransformer
    downloader.stationListFromMseed to create a station_list.json file,
    and used by other sigpig functions.

    Example: write station locations to a file
        # specify the date of interest
        date = UTCDateTime("2018-03-16T00:04:00.0Z")
        # get station locations coordinates in UTM meters easting and northing
        format = "lat_lon"
        station_locations = rattlesnake_Ridge_Station_Locations(date, format=format)

        # now write to file
        with open("station.locs", "w") as file:
            for station in station_locations.keys():
                latitude = station_locations[station][0]
                longitude = station_locations[station][1]
                line = f"{longitude} {latitude}\n"
                file.write(line)

    """
    # FIXME: needs to return time dependent station locations
    #      : check time_miner.py for station list @ time splits, make into func
    #      : get station list at specified time then get locations at that time
    #      : use locations from https://ds.iris.edu/gmap/#network=5A&maxlat=46.5301&maxlon=-120.4604&minlat=46.5217&minlon=-120.4706&drawingmode=box&planet=earth
    #      : and equivalent for UGAP
    stations = project_stations("Rattlesnake Ridge", date)
    # get all station info from IRIS GMAP2 text file
    filepath = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/stingray_rr" \
               "/gmap-stations.txt"
    loc_dict = process_gmap_file(filepath)

    # initialize location storage
    latitudes = []
    longitudes = []
    elevations = []

    # loop over stations and get locations on specified date
    for station in stations:
        # guard against missing entries
        latitude = ""
        longitude = ""
        elevation = ""

        # location_dict keys are stations
        period_dict = loc_dict[str(station)]
        # loop over period_dict keys, which are deployment end dates
        for end_date in period_dict.keys():
            # check that date is within station period
            if date < UTCDateTime(end_date):
                if date > UTCDateTime(period_dict[end_date][0]):
                    # date is within specified range, so store info
                    latitude = float(period_dict[end_date][1])
                    longitude = float(period_dict[end_date][2])
                    elevation = float(period_dict[end_date][3])

        # append info to main lists
        latitudes.append(latitude)
        longitudes.append(longitude)
        elevations.append(elevation)

    # # in decimal degrees format
    # stations = (1,2,3,4,5,6,7,8,9,10,12,13,15,16,17,18,20,21,22,23,25,26,27,
    #             28,30,31,32,33,34,35,36,37,38,39,40,41,42,'UGAP3','UGAP5',
    #             'UGAP6')
    # latitudes = (46.5283582,46.5282788,46.5282528,46.5281667,46.5281717,
    #              46.5281396,46.5281154,46.5280997,46.5280294,46.5279883,
    #              46.527869,46.5280507,46.5277918,46.5276182,46.5275217,
    #              46.5273933,46.5272481,46.5271363,46.5270128,46.5269864,
    #              46.5268652,46.5266988,46.5266946,46.5265515,46.5264383,
    #              46.5263249,46.526291,46.5261454,46.526008,46.5260073,
    #              46.5257922,46.5257573,46.5255906,46.525457,46.5253461,
    #              46.5253264,46.5251281,46.52764,46.52663,46.52662)
    # longitudes = (-120.4667914,-120.4662543,-120.4671993,-120.4663367,
    #               -120.4665884,-120.4658735,-120.4668433,-120.4661669,
    #               -120.4669906,-120.4660337,-120.4660045,-120.4663792,
    #               -120.4662038,-120.4658907,-120.4658828,-120.4657754,
    #               -120.465704,-120.4656657,-120.4652239,-120.4656266,
    #               -120.4656232,-120.465502,-120.4652094,-120.4655297,
    #               -120.4655254,-120.4651244,-120.465529,-120.4655384,
    #               -120.4650622,-120.4654893,-120.465437,-120.4649769,
    #               -120.4653985,-120.4654196,-120.4650617,-120.4654122,
    #               -120.4654584,-120.46564,-120.46553,-120.46637)
    # elevations = (435.5141526063508,459.94244464035376,419.17496055551493,
    #               456.4002279560358,445.2225853337437,471.96515712383257,
    #               433.75775906071993,463.0207557157984,427.2243674889657,
    #               466.22453723535386,465.71123190316104,454.23827459858546,
    #               459.72188069932287,464.2974124921104,462.3340642200634,
    #               459.20987810605817,455.81267908165233,452.4536132975136,
    #               449.39541344789984,447.7972434217686,444.0736248417953,
    #               440.47945136171927,441.41787035305947,436.52746672411286,
    #               433.50566973487213,433.6467649166143,430.2554423791601,
    #               425.63420243981653,426.1603861687825,421.53271734479665,
    #               416.2025875386629,421.73271284375056,411.9060515678977,
    #               407.8851035175373,412.7486362327675,404.519552729108,
    #               397.5652740413809,465.2,435.4,435.4)
    location_dict = {}

    # convert to requested format (already in decimal degrees)
    if format == "UTM" or format == "utm":
        for index, station in enumerate(stations):
            utm_conversion = utm.from_latlon(latitudes[index], longitudes[
                                                                        index])
            easting = utm_conversion[0]
            northing = utm_conversion[1]
            location_dict[str(station)] = [easting, northing, elevations[
                                                                        index]]

    # else return latitude/longitude
    else:
        for index, station in enumerate(stations):
            location_dict[str(station)] = [latitudes[index], longitudes[index],
                                           elevations[index]]

    return location_dict


def dtm_to_grid(project_name, UTM=False, stingray_format=False):
    """
    Takes a DTM file and converts it to a file that is a GMT-compatible grid
    for plotting. For more information on netCDF see:
    https://towardsdatascience.com/create-netcdf-files-with-python-1d86829127dd
    or converts the DTM file to the format necessary for Stingray (which is
    a .mat file containing the same information in a different format).

    # FIXME: only UTM implementation has been tested. Lat/lon has bugs.

    Example:
    project_name = "Rattlesnake Ridge"
    # write elevation grid in UTM coordinates
    dtm_to_netcdf(project_name, UTM=True)
    """

    if project_name == "Rattlesnake Ridge":
        # load arrays from raster file
        raster_file = '/Users/human/Dropbox/Programs/lidar/yakima_basin_2018_dtm_43.tif'

        if UTM:
            if stingray_format:
                # stingray spatial limits
                x_limits = [694.15, 694.45]
                y_limits = [5155.40, 5155.90]
            else:
                # expanded spatial limits for plotting NLL results in GMT
                x_limits = [694.10, 694.50]
                y_limits = [5155.3, 5155.99]

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
            # testing limits
            # x_limits = [-120.480, -120.462]
            # y_limits = [46.519, 46.538]

            # limits
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
                                raster_file, x_limits, y_limits, plot=False,
                                UTM=UTM)
        if stingray_format:
            # define header
            elev_header = [x_limits[0], x_limits[1], y_limits[0], y_limits[1],
                           x_step, y_step, num_x_steps, num_y_steps]

            elev_dict['data'] = np.rot90(elevation_grid, k=3)

            savemat("/Users/human/git/sigpig/sigpig/stingray/srInput"
                    "/srElevation_TN.mat",
                    {'srElevation': elev_dict})
        else:
            # save grid to NetCDF file
            filename = 'gridded_rr_dtm.nc'
            ds = nc.Dataset(filename, 'w', format='NETCDF4')
            # add no time dimension, and lat & lon dimensions
            time = ds.createDimension('time', None)
            lat = ds.createDimension('lat', num_y_steps)
            lon = ds.createDimension('lon', num_x_steps)
            # generate netCDF variables to store data
            times = ds.createVariable('time', 'f4', ('time',))
            lats = ds.createVariable('lat', 'f4', ('lat',))
            lons = ds.createVariable('lon', 'f4', ('lon',))
            value = ds.createVariable('value', 'f4', ('time', 'lat', 'lon',))
            value.units = 'm'
            # set spatial values of grid
            lats[:] = np.flip(latitude_grid[:,0])
            lons[:] = longitude_grid[0,:]
            value[0, :, :] = np.flip(elevation_grid, axis=0)
            # close the netCDF file
            ds.close()

    return None


def eqTransformer_Formatter(project_Name: str, start_Time, end_Time):
    """
    Formats data into EQTransformer station subdirectory format and naming
    convention.

    Example:
    start_Time = UTCDateTime("2018-03-13T01:31:59.0Z")
    end_Time =   UTCDateTime("2018-03-13T01:41:01.0Z")
    project_Name = "Rattlesnake Ridge"
    eqTransformer_Formatter(project_Name, start_Time, end_Time)

    """

    if project_Name == "Rattlesnake Ridge":
        # build filepath list based on dates
        node = ['DP1', 'DP2', 'DPZ'] # nodal seismometer channels
        ugap = ['EHN', 'EHE', 'EHZ']
        stations_channels = {'1': node, '2': node, '3': node, '4': node,
                             '5': node, '6': node, '7': node, '8': node,
                             '9': node, '10': node, '12': node, '13': node,
                             '15': node, '16': node, '17': node, '18': node,
                             '20': node, '21': node, '22': node, '23': node,
                             '25': node, '26': node, '27': node, '28': node,
                             '30': node, '31': node, '32': node, '33': node,
                             '34': node, '35': node, '36': node, '37': node,
                             '38': node, '39': node, '40': node, '41': node,
                             '42': node, 'UGAP3': ugap, 'UGAP5': ugap, 'UGAP6': ugap}

        for station in stations_channels:
            print(f" Processing {station}")
            # make directory based on station name
            os.mkdir(f"/Users/human/Dropbox/Programs/eqt/rr/downloads_mseeds/{station}")

            for channel in stations_channels[station]:
                filepath = f"/Volumes/newton_6TB/RR_MSEED/5A.{station}.." \
                           f"{channel}.{start_Time.year}-" \
                           f"{start_Time.month:02}-{start_Time.day:02}T00.00.00.ms"
                # load data, trim, and save file
                #obspyStream = Stream()
                obspyStream = read(filepath)

                interpolate = False
                # # interpolation breaks on UGAP6
                # for trace in obspyStream:
                #     # the sampling rate of nodes is 250 Hz
                #     if trace.stats.sampling_rate != 250.0:
                #         print(f"Trace has a different sampling rate than "
                #               f"nodes. ")
                #         # raise the flag
                #         interpolate = True
                if interpolate:
                    print("Interpolating...")
                    # interpolate to correct sampling rate and trim to correct time period
                    sampling_Rate = 250.0
                    npts = int((end_Time - start_Time) * sampling_Rate)
                    obspyStream.interpolate(sampling_Rate, method="lanczos",
                                            starttime=start_Time, npts=npts,
                                            a=30)
                    # obspy.signal.interpolation.plot_lanczos_windows(a=30)
                else:
                    # trim to specified time period
                    obspyStream = obspyStream.trim(start_Time, end_Time)

                # format filename and save Stream as miniseed file
                network = obspyStream[0].stats.network


                start_Time_Stamp = f"{obspyStream[0].stats.starttime.year}{obspyStream[0].stats.starttime.month:02}{obspyStream[0].stats.starttime.day:02}T{obspyStream[0].stats.starttime.hour:02}{obspyStream[0].stats.starttime.minute:02}{obspyStream[0].stats.starttime.second:02}Z"
                end_Time_Stamp = f"{obspyStream[0].stats.endtime.year}{obspyStream[0].stats.endtime.month:02}{obspyStream[0].stats.endtime.day:02}T{obspyStream[0].stats.endtime.hour:02}{obspyStream[0].stats.endtime.minute:02}{obspyStream[0].stats.endtime.second:02}Z"

                # writes to path
                obspyStream.write(f"/Users/human/Dropbox/Programs/eqt"
                                  f"/rr/downloads_mseeds/{station}/"
                                  f"{network}.{station}.."
                                  f"{channel}__{start_Time_Stamp}__"
                                  f"{end_Time_Stamp}.mseed", format="MSEED")

    return None


def get_trace_properties(trace, pick_time, duration):
    """ Takes in a Obspy trace object, then calculates and returns arrays
    containing the first derivative, second derivative, and curvature of the
    trace for the specified duration (in seconds) centered on the specified
    phase pick time (pick_time).

        Example:
            # use *events* as returned by data.top_n_autopicked_events

            # define the file paths containing the autopicked .mrkr file
            autopicked_file_path = "/Users/human/Dropbox/Programs/unet/autopicked_events_03_13_2018.mrkr"
            # define the desire number of events to get
            n = 500
            events = top_n_autopicked_events(autopicked_file_path, n)
            # save the dict keys (hashed event ids) into a list for easy access
            event_ids = list(events.keys())

            # select the event of interest
            event = events[event_ids[0]].copy()

            # get stream containing all phases in the event
            stream = get_event_stream(event)
            # select the trace of interest
            index = 0
            trace = stream[index]
            pick_time = event[index]['time']
            duration = 0.4 # in seconds
            dy, d2y, curvature, fits, MADs = get_trace_properties(trace, pick_time, duration)
    """
    # get trace data
    trace_data = trace.data.copy()

    # get the data evnelope (a curve outlining extremes of abs(trace.data))
    data_envelope = obspy.signal.filter.envelope(trace.data)

    # # scale the trace data to compare shape properties across traces
    # scaler = preprocessing.StandardScaler().fit(trace_data.reshape(-1, 1))
    # trace_data = scaler.transform(trace_data.reshape(-1, 1)).reshape(1, -1)[0]
    # # shift the mean to 10
    # trace_data = trace_data + 10

    # get the trace times
    trace_times_DT = np.asarray(
                  [trace.stats.starttime + offset for offset in trace.times()])
    trace_times = trace.times("matplotlib")

    # get index into trace associated with pick time
    pick_time_index = (np.abs([data_time_DT - pick_time for data_time_DT in
                               trace_times_DT])).argmin()
    sampling_rate = 250  # Hz

    # initialize lists to store transformed data
    max_pool_amplitude = []
    max_pool_times = []
    max_pool_indices = []  # this is used later for polynomial fitting

    # make indices to loop over all data in specified duration, centered on
    # the pick time
    starting_array_index = pick_time_index - int(duration / 2 * sampling_rate)
    ending_array_index = pick_time_index + int(duration / 2 * sampling_rate)

    # append all values before pick time
    for index in range(starting_array_index, pick_time_index + 1):
        max_pool_amplitude.append(abs(trace_data[index]))
        max_pool_times.append(trace_times[index])
        max_pool_indices.append(index)

    # after pick time, perform 2/4 max pooling on abs(data), so loop over
    # remaining array in chunks of 4
    for index in range(pick_time_index + 1, ending_array_index + 1, 4):
        # store a chunk of 4, data and times
        temp_data = abs(trace_data[index:index + 4])
        temp_times = trace_times[index:index + 4]
        # get indices of 2 max amplitude values from abs(data)
        top_2_indices = np.argpartition(temp_data, -2)[-2:]
        # append top 2 to the max_pool data lists
        for top_2_index in top_2_indices:
            max_pool_amplitude.append(abs(trace_data[top_2_index + index]))
            max_pool_times.append(trace_times[top_2_index + index])
            max_pool_indices.append(top_2_index + index)

    # convert max_pool_amplitude to numpy array for plotting calls
    max_pool_amplitude = np.asarray(max_pool_amplitude)

    # fit a polynomial to the max pooled data using the indices as x values
    # since polyfit hates mpl dates (small ranges)
    polynomial_degree = 12
    p = np.poly1d(
        np.polyfit(max_pool_indices, max_pool_amplitude, polynomial_degree))
    t = np.linspace(max_pool_indices[0], max_pool_indices[-1], num=1000,
                    endpoint=True)

    # transform trace envelope in same way to check its shape
    temp_xs = np.linspace(starting_array_index-10, ending_array_index+10,
                          num=120)
    temp_envelope = data_envelope[
                    starting_array_index-10:ending_array_index+10]
    f = interpolate.interp1d(temp_xs, temp_envelope, kind="cubic")

    # dy/dx first derivative, use p(t) or f(t) to choose your own adventure
    dy = np.gradient(p(t), t)
    # d2y/dx2 second derivative
    d2y = np.gradient(dy, t)
    # calculate curvature
    curvature = np.abs(d2y) / (np.sqrt(1 + dy ** 2)) ** 1.5

    # get index into max_pool amplitude corresponding to pick_time_index
    max_pool_pick_time_index = np.where(max_pool_indices ==
                                        pick_time_index)[0][0]
    # calculate median absolute deviation of three windows
    w1_median = np.median(max_pool_amplitude[:max_pool_pick_time_index - 10])
    w1_MAD = stats.median_abs_deviation(
        max_pool_amplitude[:max_pool_pick_time_index - 10])
    w2_median = np.median(max_pool_amplitude[max_pool_pick_time_index -
                                             10:max_pool_pick_time_index + 6])
    w2_MAD = stats.median_abs_deviation(max_pool_amplitude[
                                        max_pool_pick_time_index -
                                        10:max_pool_pick_time_index + 6])
    w3_median = np.median(max_pool_amplitude[max_pool_pick_time_index + 6:])
    w3_MAD = stats.median_abs_deviation(max_pool_amplitude[
                                        max_pool_pick_time_index + 6:])
    MADs = [w1_median, w1_MAD, w2_median, w2_MAD, w3_median, w3_MAD]

    # TODO: implement wavelet transforms or something else if this isn't robust

    # store fitting parameters in a list
    fits = [max_pool_indices, max_pool_amplitude, p, t, polynomial_degree]

    return dy, d2y, curvature, fits, MADs


def plot_trace_properties(trace, pick_time, duration, dy, d2y, curvature,
                          fits, MADs):
    """ Plots the trace and its properties (as returned by
    get_trace_properties) to visualize.

    Example:
        # use *events* as returned by data.top_n_autopicked_events

        # define the file paths containing the autopicked .mrkr file
        autopicked_file_path = "/Users/human/Dropbox/Programs/unet/autopicked_events_03_13_2018.mrkr"
        # define the desire number of events to get
        n = 500
        events = top_n_autopicked_events(autopicked_file_path, n)
        # save the dict keys (hashed event ids) into a list for easy access
        event_ids = list(events.keys())

        # select the event of interest
        event = events[event_ids[0]].copy()

        # get stream containing all phases in the event
        stream = get_event_stream(event)
        # select the trace of interest
        index = 0
        trace = stream[index]
        pick_time = event[index]['time']
        duration = 0.4 # in seconds
        dy, d2y, curvature, fits, MADs = get_trace_properties(trace, pick_time, duration)
        fig = plot_trace_properties(trace, pick_time, duration, dy, d2y,
                                    curvature, fits, MADs)
        fig.savefig(f"event0_index{index}", dpi=200)

        # then plot another trace's properties to compare: 0, 2, 16, 20
        indices = [2, 16, 20]
        for index in indices:
            trace = stream[index]
            pick_time = event[index]['time']
            dy, d2y, curvature, fits, MADs = get_trace_properties(trace,
                                                           pick_time, duration)
            fig = plot_trace_properties(trace, pick_time, duration, dy, d2y,
                                        curvature, fits, MADs)
            fig.savefig(f"event0_index{index}", dpi=200)
    """
    # get trace data
    trace_data = trace.data.copy()
    # # scale the trace data to compare shape properties across traces
    # scaler = preprocessing.StandardScaler().fit(trace_data.reshape(-1, 1))
    # trace_data = scaler.transform(trace_data.reshape(-1, 1)).reshape(1, -1)[0]
    # # shift the mean to 10
    # trace_data = trace_data + 10
    # get trace times
    trace_times_DT = np.asarray(
        [trace.stats.starttime + offset for offset in trace.times()])
    trace_times = trace.times("matplotlib")

    # only considering data + and - 0.2 seconds from pick time
    pick_time_index = (np.abs([data_time_DT - pick_time for data_time_DT in
                               trace_times_DT])).argmin()  # get pick time index in data array
    sampling_rate = 250  # Hz

    # make indices to loop over all data within 0.2s of pick time
    starting_array_index = pick_time_index - int(duration / 2 * sampling_rate)
    ending_array_index = pick_time_index + int(duration / 2 * sampling_rate)

    # get time series data for plotting
    temp_xs = np.linspace(starting_array_index, ending_array_index, num=100)
    temp_data = trace_data[starting_array_index:ending_array_index]

    # extract info from fits list
    max_pool_indices, max_pool_amplitude, p, t, polynomial_degree = fits
    # get index into max_pool amplitude corresponding to pick_time_index
    max_pool_pick_time_index = np.where(max_pool_indices ==
                                        pick_time_index)[0][0]

    # plot max pooling, fit, and trace data as above for reference
    fig, ax = plt.subplots(5, figsize=(8, 15))

    # plot the time series for reference
    ax[0].plot(temp_xs, temp_data, c='gray', linewidth=0.7)
    # plot the pick time
    ax[0].plot([pick_time_index, pick_time_index],
               [temp_data.min(), temp_data.max()], c="r")
    # plot max_pool transformed data
    ax[0].scatter(max_pool_indices, max_pool_amplitude, s=3, c="k")
    # plot the fit to the max_pool amplitude data
    ax[0].plot(t, p(t), c="b", linewidth=0.7)
    # set x_lim to + and - 0.2 seconds surrounding pick time
    ax[0].set_xlim([starting_array_index+5, ending_array_index-10])
    ax[0].set_title(f"max pooling polyfit d={polynomial_degree}")

    # plot zero line
    ax[1].plot([t[0], t[-1]], [0, 0], c='black', linewidth=0.7, label='zero')
    # plot first derivative
    ax[1].plot(t[50:-100], dy[50:-100], c='blue', linewidth=0.7, label='dy/dx')
    # plot the pick time
    ax[1].plot([pick_time_index, pick_time_index],
               [dy[50:-100].min() - 1, dy[50:-100].max() + 1], c="r")
    ax[1].legend()
    ax[1].set_title("1st derivative of max pooling polyfit")
    # ax[1].set_ylim([dy[5:-10].min() - 1, dy[5:-10].max() + 1])
    ax[1].set_xlim([starting_array_index+5, ending_array_index-10])

    # plot zero line
    ax[2].plot([t[0], t[-1]], [0, 0], c='black', linewidth=0.7, label='zero')
    # plot second derivative
    ax[2].plot(t[50:-100], d2y[50:-100], c='orange', linewidth=0.7, label='d2y/dx2')
    # plot the pick time
    ax[2].plot([pick_time_index, pick_time_index],
               [d2y[50:-100].min() - 1, d2y[50:-100].max() + 1], c="r")
    ax[2].legend()
    ax[2].set_title("2nd derivative of max pooling polyfit")
    # ax[2].set_ylim([d2y.min() - 1, d2y.max() + 1])
    ax[2].set_xlim([starting_array_index+5, ending_array_index-10])

    # plot the curvature
    ax[3].plot(t[50:-100], curvature[50:-100], c='green', linewidth=0.7, label='curvature')
    # plot the pick time
    ax[3].plot([pick_time_index, pick_time_index],
               [curvature[50:-100].min(), curvature[50:-100].max()], c="r")
    ax[3].legend()
    ax[3].set_title("curvature of max pooling polyfit")
    ax[3].set_xlim([starting_array_index+5, ending_array_index-10])

    # plot the time series for reference, then MAD
    ax[4].plot(temp_xs, temp_data, c='gray', linewidth=0.7)
    # plot the pick time
    ax[4].plot([pick_time_index, pick_time_index],
               [temp_data.min(), temp_data.max()], c="r")
    # plot max_pool transformed data
    ax[4].scatter(max_pool_indices, max_pool_amplitude, s=3, c="k")

    # plot +- 2 MAD of three windows
    # first window
    w1_median, w1_MAD, w2_median, w2_MAD, w3_median, w3_MAD = MADs
    ax[4].plot([max_pool_indices[0], max_pool_indices[
                max_pool_pick_time_index - 9]], [w1_median + (2*w1_MAD),
                w1_median + (2*w1_MAD)], c="c", label="w1 +-2MAD")
    ax[4].plot([max_pool_indices[0], max_pool_indices[
                max_pool_pick_time_index - 9]], [w1_median - (2*w1_MAD),
                w1_median - (2*w1_MAD)], c="c", label=None)
    ax[4].plot([max_pool_indices[0], max_pool_indices[
                max_pool_pick_time_index - 9]], [w1_median + (8*w1_MAD),
                w1_median + (8*w1_MAD)], c="c", label="w1 +-8MAD")
    ax[4].plot([max_pool_indices[0], max_pool_indices[
                max_pool_pick_time_index - 9]], [w1_median - (8*w1_MAD),
                w1_median - (8*w1_MAD)], c="c", label=None)

    # second window
    ax[4].plot([max_pool_indices[max_pool_pick_time_index - 10],
                max_pool_indices[max_pool_pick_time_index + 5]],
               [w2_median + (2 * w2_MAD), w2_median + (2 * w2_MAD)], c="g",
               label="w2 +-2MAD")
    ax[4].plot([max_pool_indices[max_pool_pick_time_index - 10],
                max_pool_indices[max_pool_pick_time_index + 5]],
               [w2_median - (2 * w2_MAD), w2_median - (2 * w2_MAD)], c="g",
               label=None)
    ax[4].plot([max_pool_indices[max_pool_pick_time_index - 10],
                max_pool_indices[max_pool_pick_time_index + 5]],
               [w2_median + (8 * w2_MAD), w2_median + (8 * w2_MAD)], c="g",
               label="w2 +-8MAD")
    ax[4].plot([max_pool_indices[max_pool_pick_time_index - 10],
                max_pool_indices[max_pool_pick_time_index + 5]],
               [w2_median - (8 * w2_MAD), w2_median - (8 * w2_MAD)], c="g",
               label=None)

    # third window
    ax[4].plot([max_pool_indices[max_pool_pick_time_index + 6],
                max_pool_indices[-1]], [w3_median + (2 * w3_MAD),
                w3_median + (2 * w3_MAD)], c="m", label="w3 +-2MAD")
    ax[4].plot([max_pool_indices[max_pool_pick_time_index + 6],
                max_pool_indices[-1]], [w3_median - (2 * w3_MAD),
                w3_median - (2 * w3_MAD)], c="m", label=None)
    ax[4].plot([max_pool_indices[max_pool_pick_time_index + 6],
                max_pool_indices[-1]], [w3_median + (8 * w3_MAD),
                w3_median + (8 * w3_MAD)], c="m", label="w3 +-8MAD")
    ax[4].plot([max_pool_indices[max_pool_pick_time_index + 6],
                max_pool_indices[-1]], [w3_median - (8 * w3_MAD),
                w3_median - (8 * w3_MAD)], c="m", label=None)

    # set x_lim to + and - 0.2 seconds surrounding pick time
    ax[4].set_xlim([starting_array_index + 5, ending_array_index - 10])
    ax[4].set_title(f"max pooling +- 2 & 8 MAD")
    ax[4].legend()

    plt.show()

    return fig


def top_n_autopicked_events(autopicked_file_path, n):
    """ Returns a dictionary containing the top n events from the specified
    mrkr file (snuffler format), ranked by the number of phases. Also writes
    the dictionary of events to a file called top_events_dict.pkl. Specify
    n = -1 to select all events.

    Example:
        # define the file paths containing the autopicked .mrkr file
        # autopicked_file_path = "/Users/human/Dropbox/Programs/unet/autopicked_test.mrkr"
        autopicked_file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/amplitude_locations/res.mrkr"
        # autopicked_file_path = "/Users/human/Dropbox/Programs/unet/autopicked_events_03_13-07_09_2018.mrkr"
        autopicked_file_path = "/Users/human/Dropbox/Programs/unet/autopicked_events_03_13_2018.mrkr"
        # define the desired number of events to get
        n = -1
        events = top_n_autopicked_events(autopicked_file_path, n)
    """
    # store the events in a structure
    events = {}

    # event tags and phase tags can be out of order so the autopicked .mrkr
    # file is looped over two times to collect events then phases.

    # build a list indicating the order of events to compare neighbors
    event_order = []
    # store events by reading the marker file line by line
    with open(autopicked_file_path, 'r') as file:
        for index, line_contents in enumerate(file):
            # only consider lines containing events
            if line_contents[0:5] == 'event':
                # store the hash ID for the event
                hash_id = line_contents.strip()[36:64]
                # generate an event entry in the dict. This storage method has
                # the side effect of ignoring duplicate events.
                events[hash_id] = []
                event_order.append(hash_id)

    # store arrival times by reading the marker file line by line
    with open(autopicked_file_path, 'r') as file:
        for index, line_contents in enumerate(file):
            if line_contents[0:5] == 'phase' and len(line_contents) < 140:
                # store the hash ID, station, and time for the phase
                hash_id = line_contents.strip()[52:80]
                station_components = line_contents.strip()[
                                     36:52].strip().split('.')
                # station format is constructed to match sta.chan
                phase_station = f"{station_components[1]}" \
                                f".{station_components[3]}"
                phase_time = UTCDateTime(line_contents.strip()[7:32])

                # check for duplicates
                duplicate = False
                if hash_id in events.keys():
                    for entry in events[hash_id]:
                        if entry['station'] == phase_station:
                            duplicate = True
                            break

                    # store the station and time of the phase in a list of dicts
                    if not duplicate:
                        events[hash_id].append({'station': phase_station, 'time':
                            phase_time})

    # Now we have a dict where each key is a unique hash id for an event, and
    # each entry is a list of dicts containing time of the first arrival and
    # the station it was recorded on.

    # There are some events with different event id's but identical phase
    # picks. Duplicate phase times within + or - the phase_time_threshold are
    # deleted in the loop below.
    phase_time_threshold = 0.1 # in seconds
    deleted_phase_count = 0
    event_ids = list(events.keys())

    # # This traversal is not optimized; it is O(n^2) so it's slowww; ~10000
    # # day runtime for 4.8 million events so that isn't going to work.
    # # loop over each event and display a progress bar
    # for index in tqdm(range(len(event_ids))):
    #     event_id = event_ids[index]
    #
    #     # loop over each phase in each event
    #     for phase in events[event_id]:
    #         # compare this phase with every other phase
    #         for comp_event_id in events.keys():
    #             # only compare selected phase to phases with other event id's
    #             if event_id != comp_event_id:
    #                 # get the length of the list storing phases and loop
    #                 # over it in reverse to accommodate deletion of items
    #                 for phase_index in range(len(events[comp_event_id]) - 1,
    #                                          -1, -1):
    #                     comp_phase = events[comp_event_id][phase_index]
    #                     # compare the phase station and time
    #                     if (phase['station'] == comp_phase['station']) and \
    #                        (phase['time'] >= (comp_phase['time'] -
    #                        phase_time_threshold)) and  (phase['time'] <= (
    #                        comp_phase['time'] + phase_time_threshold)):
    #                         # remove the comparison phase from the list
    #                         del events[comp_event_id][phase_index]
    #                         deleted_phase_count += 1

    # # Optimized traversal of every event to compare and delete duplicate phase
    # # time arrivals belonging to different events. Phases times in
    # # different events are only compared if the events are within 1 minute
    # # of each other, in which case every phase for each phase is compared.
    # # Runtime still too slow at 2480 days per 4.8 million events. Nope.
    # # build a list of event times from the first phase time so they aren't
    # # looked up each time
    # event_times = {event_id: events[event_id][0]["time"] for event_id in
    #                events.keys()}
    #
    # # loop over each event and display a progress bar
    # for index in tqdm(range(len(event_ids))):
    #     event_id = event_ids[index]
    #
    #     # loop over every event again so each event is compared to every event
    #     for comp_event_id in events.keys():
    #         # avoid comparing an event to itself
    #         if event_id != comp_event_id:
    #             # get the event time and compare it to the other event
    #             if abs(event_times[event_id] - event_times[comp_event_id]) <60:
    #                 # loop over each phase in the original event (outermost
    #                 # loop)
    #                 for phase in events[event_id]:
    #                     # get the length of the list storing phases and loop
    #                     # over it in reverse to accommodate deletion of items
    #                     for phase_index in range(len(events[comp_event_id])
    #                                              - 1, -1, -1):
    #                         comp_phase = events[comp_event_id][phase_index]
    #                         # compare the phase station and time
    #                         if (phase['station'] == comp_phase['station']) and \
    #                                 (phase['time'] >= (comp_phase['time'] -
    #                                                    phase_time_threshold)) and (
    #                                 phase['time'] <= (
    #                                 comp_phase[
    #                                     'time'] + phase_time_threshold)):
    #                             # remove the comparison phase from the list
    #                             del events[comp_event_id][phase_index]
    #                             deleted_phase_count += 1
    #
    #                             # stop searching for duplicate after it's found
    #                             break

    # Another optimization using event neighbors rather than comparing each
    # event time. This loop takes ## minutes to cull 4.8 million events.
    # build a list of event times from the first phase time so they aren't
    # looked up each time
    event_times = {event_id: events[event_id][0]["time"] for event_id in
                   events.keys()}

    # loop over each event from the in-order event list, compare it to 3
    # events on each side,  and display a progress bar
    num_events = len(event_order)
    for index in tqdm(range(num_events)):
        event_id = event_order[index]

        # case: first three events
        if index < 3:
            # loop over any events before and the 3 events after current
            for comp_event_index in range(0, index + 4):
                comp_event_id = event_order[comp_event_index]
                # avoid comparing an event to itself
                if event_id != comp_event_id:
                    # get the event time and compare it to the other event
                    if abs(event_times[event_id] - event_times[
                        comp_event_id]) < 60:
                        # loop over each phase in the original event (outermost
                        # loop)
                        for phase in events[event_id]:
                            # get the length of the list storing phases and loop
                            # over it in reverse to accommodate deletion of items
                            for phase_index in range(len(events[comp_event_id])
                                                     - 1, -1, -1):
                                comp_phase = events[comp_event_id][phase_index]
                                # compare the phase station and time
                                if (phase['station'] == comp_phase[
                                    'station']) and \
                                        (phase['time'] >= (comp_phase['time'] -
                                                           phase_time_threshold)) and (
                                        phase['time'] <= (
                                        comp_phase[
                                            'time'] + phase_time_threshold)):
                                    # remove the comparison phase from the list
                                    del events[comp_event_id][phase_index]
                                    deleted_phase_count += 1

                                    # stop searching for duplicate after it's found
                                    break

        # case: last three events
        elif num_events - index <= 3:
            # loop over any events before and the 3 events after current
            for comp_event_index in range(index - 3, num_events):
                comp_event_id = event_order[comp_event_index]
                # avoid comparing an event to itself
                if event_id != comp_event_id:
                    # get the event time and compare it to the other event
                    if abs(event_times[event_id] - event_times[
                        comp_event_id]) < 60:
                        # loop over each phase in the original event (outermost
                        # loop)
                        for phase in events[event_id]:
                            # get the length of the list storing phases and loop
                            # over it in reverse to accommodate deletion of items
                            for phase_index in range(len(events[comp_event_id])
                                                     - 1, -1, -1):
                                comp_phase = events[comp_event_id][phase_index]
                                # compare the phase station and time
                                if (phase['station'] == comp_phase[
                                    'station']) and \
                                        (phase['time'] >= (comp_phase['time'] -
                                                           phase_time_threshold)) and (
                                        phase['time'] <= (
                                        comp_phase[
                                            'time'] + phase_time_threshold)):
                                    # remove the comparison phase from the list
                                    del events[comp_event_id][phase_index]
                                    deleted_phase_count += 1

                                    # stop searching for duplicate after it's found
                                    break

        # case: all other events
        else:
            # loop over the 3 events before and the 3 events after current
            for comp_event_index in range(index - 3, index + 4):
                comp_event_id = event_order[comp_event_index]
                # avoid comparing an event to itself
                if event_id != comp_event_id:
                    # get the event time and compare it to the other event
                    if abs(event_times[event_id] - event_times[
                        comp_event_id]) < 60:
                        # loop over each phase in the original event (outermost
                        # loop)
                        for phase in events[event_id]:
                            # get the length of the list storing phases and loop
                            # over it in reverse to accommodate deletion of items
                            for phase_index in range(len(events[comp_event_id])
                                                     - 1, -1, -1):
                                comp_phase = events[comp_event_id][phase_index]
                                # compare the phase station and time
                                if (phase['station'] == comp_phase['station']) and \
                                        (phase['time'] >= (comp_phase['time'] -
                                                           phase_time_threshold)) and (
                                        phase['time'] <= (
                                        comp_phase[
                                            'time'] + phase_time_threshold)):
                                    # remove the comparison phase from the list
                                    del events[comp_event_id][phase_index]
                                    deleted_phase_count += 1

                                    # stop searching for duplicate after it's found
                                    break

    print(f"Deleted {deleted_phase_count} duplicate phases.")
    # now delete events that have had all phases removed or < 4 phases
    deletion_keys = []
    for event_id in events.keys():
        if len(events[event_id]) < 4:
            deletion_keys.append(event_id)
    [events.pop(deletion_key, None) for deletion_key in deletion_keys]

    # make lists of event ids and the number of phases for each event
    event_ids = []
    num_phases = []
    for event_id in events.keys():
        event_ids.append(event_id)
        num_phases.append(len(events[event_id]))

    # get indices for sorted num_phases array
    sort_indices = np.argsort(num_phases)

    # if n=-1, reset n to number of events
    if n == -1:
        n = sort_indices.size

    outfile = open(f"{n}_events_dict.pkl", 'wb')
    pickle.dump(events, outfile)
    outfile.close()
    outfile = open(f"{n}_events_order.pkl", 'wb')
    pickle.dump(event_order, outfile)
    outfile.close()

    # find the top n events with most phases
    top_n_sort_indices = [sort_indices[index] for index in range(-1, (n + 1)
                                                                 * -1, -1)]
    num_phases = np.asarray(num_phases)
    event_ids = np.asarray(event_ids)
    num_phases = num_phases[top_n_sort_indices]
    event_ids = event_ids[top_n_sort_indices]

    # save pickle file with top n events
    top_events = {event_id: events[event_id] for event_id in event_ids}
    outfile = open(f"top_{n}_events_dict.pkl", 'wb')
    pickle.dump(top_events, outfile)
    outfile.close()

    return top_events


def get_event_stream(event):
    """ Takes in an event from an events dict (as returned by
    top_n_autopicked_events function) and returns a stream
    containing traces corresponding to the phases of the event.

    Example:
        # define the file paths containing the autopicked .mrkr file
        autopicked_file_path = "/Users/human/Dropbox/Programs/unet/autopicked_events_03_13_2018.mrkr"
        # define the desired number of events to get
        n = 100
        events = top_n_autopicked_events(autopicked_file_path, n)

        # specify the event of interest from *events* as returned by
        # top_n_autopicked_events function

        # loop over events and write miniseed file of all traces for each pick
        for i in events.keys():
            event = events[i].copy()
            start_Time = event[0]['time'] - 0.5
            end_Time = start_Time + 1
            project_Name = "Rattlesnake Ridge"
            channels = ['DP1', 'EHN']
            trim_Daily_Waveforms(project_Name, start_Time, end_Time, channels, write_File=True)

        event = events['FA1UKxKJjSEZ-fpEj5IiLWMZd2I='].copy()
        event = events['LHRcBsa7u2QKVQ0BmOJksKt2DSA='].copy()
        event = events['FOAjuz85HUiGqBliIQrLAw7HpPc='].copy()

        # get stream containing all phases in the event
        stream = get_event_stream(event)
        # plot them
        from figures import plot_event_picks
        plot_event_picks(event, plot_curvature=False)
        plt.show()
    """
    # initialize a stream to store traces
    event_stream = Stream()

    # loop over each phase in the event, where event is a list of dicts
    for index, phase in enumerate(event):
        # get the phase pick time
        phase_time = phase['time']
        # build the trace filepath
        station_components = phase['station'].split('.')
        # station format constructed to match filenames sta..chan
        phase_station = f"{station_components[0]}" \
                        f"..{station_components[1]}"

        # trace_file_prefix = '/Volumes/newton_6TB/RR_MSEED/'
        trace_file_prefix = '/Users/human/Desktop/RR_MSEED/'
        trace_file_path = f"5A.{phase_station}." \
                          f"{phase_time.year}-" \
                          f"{phase_time.month:02}-" \
                          f"{phase_time.day:02}T00.00.00.ms"
        # load the trace
        st = read(trace_file_prefix + trace_file_path)

        # get 1 minute of data to interpolate, de-trend, filter, and trim
        st.trim(phase_time - 30, phase_time + 30, pad=True,
                fill_value=0, nearest_sample=True)
        st.interpolate(sampling_rate=250.0)
        # detrend then bandpass filter
        st.detrend()
        st.filter("bandpass", freqmin=20, freqmax=60, corners=4)

        # only consider 1.5 seconds of data (this is a busy dataset)
        st.trim(phase_time - 0.6, phase_time + 0.9, pad=True,
                fill_value=0, nearest_sample=False)

        # add the trace to the figure
        trace = st[0].copy()
        # save the trace to the main stream for future work
        event_stream += trace

    return event_stream

def get_network_stream(event):
    """ Takes in an event from an events dict (as returned by
    top_n_autopicked_events function) and returns a stream containing traces
    corresponding to all station in the network at the time of the event.
    This function is used in the calculate_magnitude function.

    Example:
        # define the file paths containing the autopicked .mrkr file
        autopicked_file_path = "/Users/human/Dropbox/Programs/unet/autopicked_events_03_13_2018.mrkr"
        # define the desired number of events to get
        n = 100
        events = top_n_autopicked_events(autopicked_file_path, n)

        # define an event
        # TODO:

        # get stream containing all phases in the event
        stream = get_event_stream(event)
        # plot them
        from figures import plot_event_picks
        plot_event_picks(event, plot_curvature=False)
        plt.show()
    """
    # initialize a stream to store traces
    network_stream = Stream()

    # set a reference phase to define time of interest
    reference_phase = event[0]
    # get the phase pick time
    phase_time = reference_phase['time']

    stations = project_stations("Rattlesnake Ridge", phase_time)

    for station in stations:

        # build the trace filepath
        if isinstance(station, int):
            channel = "DP1"
        else:
            channel = "EHN"
        # station format constructed to match filenames sta..chan
        phase_station = f"{station}..{channel}"

        # trace_file_prefix = '/Volumes/newton_6TB/RR_MSEED/'
        trace_file_prefix = '/Users/human/Desktop/RR_MSEED/'
        trace_file_path = f"5A.{phase_station}." \
                          f"{phase_time.year}-" \
                          f"{phase_time.month:02}-" \
                          f"{phase_time.day:02}T00.00.00.ms"
        # load the trace
        st = read(trace_file_prefix + trace_file_path)

        # get 1 minute of data to interpolate, de-trend, filter, and trim
        st.trim(phase_time - 30, phase_time + 30, pad=True,
                fill_value=0, nearest_sample=True)
        st.interpolate(sampling_rate=250.0)
        # detrend
        st.detrend()
        # st.filter("bandpass", freqmin=20, freqmax=60, corners=4)

        # only consider 1.5 seconds of data (this is a busy dataset)
        st.trim(phase_time - 0.6, phase_time + 0.9, pad=True,
                fill_value=0, nearest_sample=False)

        # add the trace to the figure
        trace = st[0].copy()
        # save the trace to the main stream for future work
        network_stream += trace

    return network_stream

def events_dict_to_snuffler(events: dict):
    """ Takes in a dict of events as returned by the top_n_autopicked_events
    function and writes each event and its phase picks to a snuffler-type
    marker file, e.g.:
    https://github.com/pyrocko/pyrocko/blob/4971d5e5ac3c4ac4b75992452a934e01a784ea8d/src/gui/marker.py

    Example:
        # define the file paths containing the autopicked .mrkr file
        autopicked_file_path = "/Users/human/Dropbox/Programs/unet/autopicked_events_03_13_2018.mrkr"
        # define the desired number of events to get
        n = 500
        events = top_n_autopicked_events(autopicked_file_path, n)

        # write the events to a snuffler format file
        events_dict_to_snuffler(events)
    """
    # initialize list to store content to write to file
    content_list = []

    # get RR stations for station-distance hack (for snuffler plotting/picking)
    stas = project_stations("Rattlesnake Ridge", events[list(events.keys())[
                            0]][0]['time'])
    stations = {}
    for station in stas:
        stations[str(station)] = []
    # station "distance" along scarp (it's actually station order along scarp)
    station_distance = {station: index for index, station in
                        enumerate(stations)}

    # loop through each cluster label except noise cluster
    for event in events.keys():
        # define the event time based on first phase time (arbitrary)
        event_time = events[event][0]['time'].datetime.strftime('%Y-%m-%d '
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

        # loop through each pick in the event
        for index, pick in enumerate(events[event]):
            # get pick station and station distance order hack
            pick_station = pick['station'].split('.')[0]
            pick_station_hack = station_distance[pick_station]
            # get pick channel
            pick_channel = pick['station'].split('.')[1]

            # get pick time
            pick_time = pick['time'].datetime.strftime('%Y-%m-%d %H:%M:%S.%f')
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
    f = open('res.mrkr', "a")
    f.writelines(content_list)
    f.close()

    return None


def get_picked_uncertainties():
    """ Reads in a snuffler format file containing event uncertainties and
    returns the uncertainties and the waveform properties [from refactoring
    of function below]

    Example:


    """
    ...

# TODO: working here and below
#   =
#   =
#   =
#   =
#   =
#   =
#   =
#   =
#   =
def process_autopicked_events(autopicked_file_path, uncertainty_file_path):
    """ Reads snuffler format file containing autopicked & associated events
    then assigns uncertainties to the events based on the SNR, derived from
    manual uncertainty assignments. Returns a sorted dict?  # TODO:

    Example:
        # Define the file paths containing the autopicked .mrkr file and the
        # .mrkr file containing manually assigned uncertainties.
        autopicked_file_path = "/Users/human/Dropbox/Programs/unet/autopicked_events_03_13-07_09_2018.mrkr"
        uncertainty_file_path = "/Users/human/Dropbox/Programs/snuffler/214_loc_picks.mrkr"
        process_autopicked_events(autopicked_file_path, uncertainty_file_path)
    """
    # get all events (n=-1) from autopicked file
    events = top_n_autopicked_events(autopicked_file_path, -1)

    # Now we have a dict where each key is a unique hash id for an event, and
    # each entry is a list of dicts containing time of the first arrival and
    # the station it was recorded on. The next step is to build a relationship
    # between manually assigned first arrival time uncertainties and the SNR of
    # the trace to automatically assign picking uncertainties for all traces
    # based on their SNR.

    # store arrival time uncertainties by reading the marker file line by line
    uncertainties = []
    snrs = []
    shapes = []
    max_values = []
    unet_predictions = []

    # load a tensorflow model to get pick time predictions
    # build tensorflow unet model & get predictions
    model = build_unet_model()

    with open(uncertainty_file_path, 'r') as file:
        for index, line_contents in enumerate(file):
            # uncertainty lines start with 'phase'
            if line_contents[0:5] == 'phase':
                # and the date is in a unique position
                if line_contents[33:37] == '2018':
                    # store the hash id of the event
                    hash_id = line_contents.strip()[-76:-48]

                    # store the 1 sigma uncertainty of the phase arrival
                    start_time = UTCDateTime(line_contents[7:32])
                    end_time = UTCDateTime(line_contents[33:58])
                    one_sigma = (end_time - start_time) / 2
                    uncertainties.append(one_sigma)

                    # build the trace filepath
                    station_components = line_contents.strip()[
                                         -92:-76].strip().split('.')
                    # station format constructed to match filenames sta..chan
                    phase_station = f"{station_components[1]}" \
                                    f"..{station_components[3]}"
                    # trace_file_prefix = '/Volumes/newton_6TB/RR_MSEED/'
                    trace_file_prefix = '/Users/human/Desktop/RR_MSEED/'
                    trace_file_path = f"5A.{phase_station}." \
                                      f"{start_time.year}-" \
                                      f"{start_time.month:02}-" \
                                      f"{start_time.day:02}T00.00.00.ms"
                    # load the trace to calculate its SNR
                    st = read(trace_file_prefix + trace_file_path)
                    st.trim(start_time - 30, start_time + 30, pad=True,
                            fill_value=0, nearest_sample=True)
                    st.interpolate(sampling_rate=250.0)
                    st.detrend()
                    st.filter("bandpass", freqmin=20, freqmax=60, corners=4)

                    # get unet prediction from trace and pick time
                    pick_time = start_time + ((end_time - start_time) / 2)
                    unet_prediction = trace_arrival_prediction(st[0].copy(),
                                                              pick_time, model)

                    # the pick time corresponds with the middle entry in the
                    # unet predictions array @ index 60, sum over 5 samples
                    unet_predictions.append(np.sum(unet_prediction[0][58:63]))

                    # only consider 0.5 second of data (this is a busy dataset)
                    st.trim(start_time - 0.4, start_time + 0.6, pad=True,
                            fill_value=0, nearest_sample=True)

                    # calculate the SNR of the trace and store it
                    trace_snr = snr(st[0])[0]
                    snrs.append(trace_snr)
                    # get properties of the trace using 0.4 s duration
                    duration = 0.4
                    dy, d2y, curvature, fits, MADs = get_trace_properties(
                                                   st[0], pick_time, duration)
                    # store the shape properties at the pick time
                    shapes.append([dy[500], d2y[500], curvature[500]])

                    # get a subset of the trace +- 4 samples from the pick time
                    st.trim(start_time - (4/250), start_time + (4/250),
                            pad=True, fill_value=0, nearest_sample=True)
                    max_values.append(max_amplitude(st[0])[0])

    # plot waveform properties in a function -> TODO
    # make a figure showing trace properties
    gs = pplt.GridSpec(nrows=3, ncols=2)
    fig = pplt.figure(refwidth=2.2, span=False, share=False, tight=False)
    # plot uncertainties and snrs. There is not a linear relationship w/ SNR.
    ax = fig.subplot(gs[0], title=f'1σ Uncertainty vs. SNR (n={len(snrs)})',
                     xlabel='SNR @ 1.0 s', ylabel='Uncertainty (seconds)')
    # ax.format(suptitle='Properties of trace at arrival time')
    ax.scatter(snrs, uncertainties, c=unet_predictions, colorbar='ur',
               colorbar_kw={'label': 'unet prediction'}, markersize=2)

    # plot uncertainties and 1st derivatives of traces @ pick time
    shapes = np.asarray(shapes)
    ax = fig.subplot(gs[1], title=f'1σ Uncertainty vs. 1st derivative',
                     xlabel='dy/dx of traces at pick time',
                     ylabel='Uncertainty (seconds)')
    ax.scatter(shapes[:, 0], uncertainties, c=snrs, markersize=2)
    ax.set_xlim([-10, 120])

    # plot uncertainties and 2nd derivatives of traces @ pick time
    ax = fig.subplot(gs[2], title=f'1σ Uncertainty vs. 2nd derivative',
                     xlabel='d2y/dx2 of traces at pick time',
                     ylabel='Uncertainty (seconds)')
    ax.scatter(shapes[:, 1], uncertainties, c=snrs, markersize=2)
    ax.set_xlim([-10, 80])

    # plot uncertainties and curvature of traces @ pick time
    ax = fig.subplot(gs[3], title=f'1σ Uncertainty vs. curvature',
                     xlabel='curvature of traces at pick time',
                     ylabel='Uncertainty (seconds)')
    m = ax.scatter(shapes[:, 2], uncertainties, c=snrs, markersize=2)
    ax.set_xlim([-0.1, 1.0])

    # plot uncertainties and unet prediction of traces @ pick time
    ax = fig.subplot(gs[4], title=f'1σ Uncertainty vs. unet prediction',
                     xlabel='unet prediction of traces at pick time',
                     ylabel='Uncertainty (seconds)')
    m = ax.scatter(unet_predictions, uncertainties,
                   c=snrs, markersize=2)
    # ax.set_xlim([-0.1, 1.0])
    ax.colorbar(m, loc='b', locator=1, label='SNR')

    # plot uncertainties and curvature of traces @ pick time
    ax = fig.subplot(gs[5], title=f'1σ uncert. vs. unet predictions * '
                                  f'SNR',
                     xlabel='unet prediction * SNR',
                     ylabel='Uncertainty (seconds)')
    ax.scatter(np.array(snrs) * np.array(unet_predictions),
               uncertainties, c=snrs, markersize=2)

    fig.savefig(f"waveform_properties.png", dpi=200)
    fig.show()

    # TODO:
    #      fit a model to map from _something_ to uncertainty
    ...

    # use the model fit to calculate uncertainties for all autopicked events
    for event_id in events.keys():
        for index, pick in enumerate(events[event_id]):
            # get [some measure, not SNR] of pick trace

            # assign uncertainty based on [some measure, not SNR]

            # add uncertainty to phase dict
            # events[event_id][index]['uncertainty'] =

            ...

    # find the top 500ish events with most phases
    event_ids = []
    num_phases = []
    for event_id in events.keys():
        event_ids.append(event_id)
        num_phases.append(len(events[event_id]))

    num_phases = np.asarray(num_phases)
    event_ids = np.asarray(event_ids)
    top_500_phases = np.where(num_phases > 22) # pick # for top 500ish phases
    num_phases = num_phases[top_500_phases]
    event_ids = event_ids[top_500_phases]

    # save pickle file with top 430 events w/ > 22 phases
    top_events = {event_id: events[event_id] for event_id in event_ids}
    outfile = open(f"top_events_dict.pkl", 'wb')
    pickle.dump(top_events, outfile)
    outfile.close()

    return None

def get_response_files(station_list):
    """ Downloads the response files for the specified stations.

    Example:
        from time_miner import project_stations
        date = UTCDateTime("2018-03-13T00:04:00.0Z")
        station_list = project_stations("Rattlesnake Ridge", date)
        get_response_files(station_list)
    """

    datacentre = "IRIS"
    client = Client(datacentre)
    starttime = UTCDateTime("2018-03-13T00:00:00.0Z")
    endtime = UTCDateTime("2018-03-13T23:59:59.99999999Z")
    for station in station_list:
        inv = Inventory()

        if isinstance(station, int):
            try:
                inv += client.get_stations(network="5A", station=station,
                                           channel="DP1",
                                           starttime=starttime, endtime=endtime,
                                           level="response")
                inv.write(f"RESP.5A.{station}..DP1", format="STATIONXML")

            except Exception:
                print(f"failed inventory at station {station}")
                pass

        else:
            try:
                inv += client.get_stations(network="UW", station=station,
                                           channel="EHN",
                                           starttime=starttime, endtime=endtime,
                                           level="response")
                inv.write(f"RESP.UW.{station}..EHN", format="STATIONXML")

            except Exception:
                print(f"failed inventory at station {station}")
                pass

    return None

def instantaneous_frequency(trace, plots=0):
    """
    Calculates the instantaneous frequency of a time series trace.

    Example:

        # get a dictionary of top 500 events (by phase count)
        infile = open('top_events_dict.pkl', 'rb')
        events_dict = pickle.load(infile)
        infile.close()

        # get the keys from the dict to loop over
        keys = list(events_dict.keys())

        # loop over all events and get inst. freq. of every phase
        freqs = []
        for key in keys:
            event = events_dict[key]

            # get stream containing all phases in the event
            stream = get_event_stream(event)

            # # plot them
            # from figures import plot_event_picks
            # plot_event_picks(event, plot_curvature=False)
            # plt.show()

            for trace in stream:
                freqs.append(instantaneous_frequency(trace, plots=0))

        # plot the instantaneous frequency distribution
        from figures import plot_distribution
        plot_distribution(freqs, title="Instantaneous frequency distribution for 214 events", save=True)

    Another Example:
        # get a dictionary of 214 verified and QC'd events
        infile = open('214_events_dict.pkl', 'rb')
        events_dict = pickle.load(infile)
        infile.close()

        # get the keys from the dict for traversing the dict
        keys = list(events_dict.keys())

        # loop over all events and get inst. freq. of every phase
        event_streams = {}
        event_freqs = {}
        for key_idx in tqdm(range(len(keys))):
            key = keys[key_idx]
            freqs = []
            event = events_dict[key]

            # get stream containing all phases in the event
            stream = get_event_stream(event)

            # # plot them
            # from figures import plot_event_picks
            # plot_event_picks(event, plot_curvature=False)
            # plt.show()

            # save the stream to a dict
            # event_streams[key] = stream.copy()

            # get the inst. freq. of each phase pick in this event
            for trace in stream:
                freqs.append(instantaneous_frequency(trace, plots=0))

            # append all phase inst. freqs. to event inst. freq. dict
            event_freqs[key] = freqs

        # save the inst. frequency dict and stream dict to pkl files
        outfile = open(f"214_event_freqs_dict.pkl", 'wb')
        pickle.dump(event_freqs, outfile)
        outfile.close()
        # outfile = open(f"event_streams_dict.pkl", 'wb')
        # pickle.dump(event_streams, outfile)
        # outfile.close()
    """
    # store the time step duration
    dt = trace.stats.delta
    halfwin = 0.06
    # results at different halfwin
    # 0.20    29.38
    # 0.06    32.35
    # 0.02    31.47
    # 0.001   36.61
    pts = halfwin / dt
    ind = np.where(abs(trace.data) == max(abs(trace.data)))[0][0]
    if ind - pts < 1:
        out = np.nan
    elif ind + pts > len(trace):
        out = np.nan
    else:
        tmp = hilbert(trace.data[int(ind - pts) : int(ind + pts)])
        v = tmp.real
        h = tmp.imag
        dv = np.gradient(v, dt)
        dh = np.gradient(h, dt)
        f_t = 1 / (2 * np.pi) * ((v*dh - h*dv) / (v**2 + h**2))
        out = np.median(f_t)

    if plots:
        fig = plt.figure(figsize=(12, 8))
        times = trace.times()
        plt.subplot(211)
        plt.plot(times, trace.data)
        plt.plot(times[int(ind - pts) : int(ind + pts)], trace.data[int(ind
                                                      - pts) : int(ind + pts)])
        plt.subplot(212)
        plt.plot(np.linspace(0, len(v)-1, len(v)), v, 'r')
        plt.plot(np.linspace(0, len(f_t)-1, len(f_t)), f_t, 'b')
        plt.show()

    return out


def inst_freq_binning():
    """
    #TODO: write docstring

    Example:

    """
    # load dict constaining instaneous frequency of events
    # infile = open('03_13_18_event_freqs_dict.pkl', 'rb')
    infile = open('214_event_freqs_dict.pkl', 'rb')
    freqs_dict = pickle.load(infile)
    infile.close()

    # load locations of events
    # TODO: load from 03_13_18.locs files
    event_locs = pd.read_csv('res_214_A0-50_1020ms.locs')
    event_freqs = []

    # loop over all events with locations
    for id in event_locs.ID:

        # get and store the average inst. freq. of the phases as the event i.f.
        event_freqs_array = np.asarray(freqs_dict[id])
        event_freqs.append(np.nanmedian(event_freqs_array))

    # add event_freqs to event_locs dataframe
    event_locs['freq'] = event_freqs

    # bin the insantaneous frequency measurements by 10 m in Y coordinate
    y_limits = [5155300, 5155990]
    y_spacing = 10  # meters
    # y_steps = np.linspace(y_limits[0], y_limits[1], (y_limits[1] - y_limits[
    #           0] + 1) // y_spacing)
    #
    # # bin by latitude
    # counts = pd.cut(event_locs['y'], bins=y_steps).value_counts()

    # store

    # TODO: loop over search bins with dataframe querying
        # how to bin data in dataframe efficiently?

    # TODO: plot in distance along scarp?

    return inst_freq_bins


def roughness_binning():
    """
    #TODO: write docstring

    Returns:

    """
    # load point cloud with geometry statistics from csv
    # FIXME: change to full scarp.csv file
    roughness = pd.read_csv('/Users/human/Dropbox/Research/Rattlesnake_Ridge'
                            '/data/lidar/scarp_subset.csv')

    # bin the roughness measurements along 10 m of Y coordinate
    y_limits = [5155300, 5155990]
    y_steps = np.linspace(y_limits[0], y_limits[1],
                          y_limits[1] - y_limits[0] + 1)
    y_stride = 10 # meters

    # TODO: loop over search bins with dataframe querying
    for step in range(len(y_steps) // y_stride):
        # TODO: how to bin data in dataframe efficiently?
        if step == 0:
            y_step_limits = [y_steps[0], y_steps[y_stride]]
        else:
            y_step_limits = [y_steps[step * y_stride], y_steps[step *
                                                               y_stride +
                                                               y_stride]]

        # get data from dataframe that is in this bin
        # TODO: how to best do this?
        in_bin = roughness.iloc[roughness["Y"] > y_step_limits[0] and
                                roughness["Y"] <= y_step_limits[1]]

        # TODO: save some calculation from the binned data
        #     : like average roughness scaled by z extent of data?

    return roughness_bins


def compare_inst_freq_and_roughness():
    """
    # TODO: write docstring

    Example:

    """
    # get binned instantaneous frequency data
    inst_freq_bins = inst_freq_binning()

    # get binned roughness data
    roughness_bins = roughness_binning()

    # check for correlation between series
    # TODO:

    # plot inst. freq. and roughness series
    # TODO:


def calculate_magnitude():
    """ Function to generate magnitude estimates for the Rattlesnake
    Ridge dataset, calculated from EQcorrscan's singular-value decomposition
    relative moment estimation (
    https://eqcorrscan.readthedocs.io/en/latest/tutorials/mag-calc.html?highlight=similar_events_processed)

    Example:
        magnitudes = calculate_magnitude()

        from figures import plot_moment_distribution
        plot_moment_distribution(all_magnitudes, title="Relative moment distribution for top 500 events", save=True)

    """
    # define the file paths containing the autopicked .mrkr file
    autopicked_file_path = "/Users/human/Dropbox/Programs/unet/autopicked_events_03_13_2018.mrkr"
    # define the desired number of events to get
    n = 3000
    events = top_n_autopicked_events(autopicked_file_path, n)

    relative_moments = []

    # loop over events and calculate magnitudes
    event_keys = list(events.keys())
    print(f"There are {len(event_keys)} total events.")

    # > 300 breaks svd_moments function, so split mag calc in chunks of 300
    for start_index in range(0, len(event_keys), 300):
        # set the end index for this chunk
        end_index = start_index + 300
        if end_index > len(event_keys):
            end_index = len(event_keys)

        # loop over all items in this chunk
        for index in tqdm(range(start_index, end_index)):
            # make an empty list to store the streams for each event
            stream_list = []
            event_list = []
            # get stream containing all phases in the event
            event = events[event_keys[index]]
            stream = get_network_stream(event)

            # # plot them
            # from figures import plot_event_picks
            # plot_event_picks(event, plot_curvature=False)
            # plt.show()

            # store the stream and add indices to index matrix needed by EQcorrscan
            stream_list.append(stream)
            index_list = []
            for trace in stream:
                if len(trace) != 375:
                    trace.trim(trace.stats.starttime, trace.stats.starttime + (
                        374 * trace.stats.delta))

                index_list.append(index)
            event_list.append(index_list)

        event_list = np.asarray(event_list).T.tolist()
        SVectors, SValues, Uvectors, stachans = svd(stream_list=stream_list)
        rel_moments, events_out = svd_moments(u=Uvectors, s=SValues, v=SVectors,
                                    stachans=stachans, event_list=event_list)

        for rel_moment in rel_moments:
            relative_moments.append(rel_moment)

    return relative_moments
