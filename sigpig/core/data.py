"""
Functions to fetch data. 
"""

import obspy
from obspy import read, Stream
from obspy.core.utcdatetime import UTCDateTime
from obspy.clients.fdsn import Client
import calendar
from datetime import datetime
import glob
import numpy as np
import utm
from obspy import read, Stream
from obspy.core.utcdatetime import UTCDateTime
import os
from lidar import grids_from_raster

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
        traceMax = [np.max(np.abs(timeSeries[trace].data)) for trace in
                    range(len(timeSeries))]
        # return max value among all traces
        return np.max(traceMax), np.argmax(traceMax)

    # for Trace objects
    elif isinstance(timeSeries, obspy.core.trace.Trace):
        if len(timeSeries.data) > 0:
            traceMax = np.max(np.abs(timeSeries.data))
            max_index = np.argmax(np.abs(timeSeries.data))
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
    start_Time = UTCDateTime("2018-03-13T00:12:50.0Z")
    end_Time =   UTCDateTime("2018-03-13T00:16:00.0Z")
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
        files_path = "/Users/human/Dropbox/Research/Alaska/build_templates" \
                     "/2016-09-26"
        filepaths = glob.glob(f"{files_path}/*.ms")

        obspyStream = Stream()
        for filepath_idx in range(len(filepaths)):
            obspyStream += read(filepaths[filepath_idx]).merge(method=1,
                                                               fill_value=None)

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
        obspyStream = obspyStream.trim(start_Time, end_Time)

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
        format = "UTM"
        station_locations = rattlesnake_Ridge_Station_Locations(date, format=format)

        # now write to file
        with open("station.locs", "w") as file:
            for station in station_locations.keys():
                easting = station_locations[station][0]
                northing = station_locations[station][1]
                line = f"{station},{easting},{northing}\n"
                file.write(line)

    """
    # FIXME: needs to return time dependent station locations
    #      : check time_miner.py for station list @ time splits, make into func
    #      : get station list at specified time then get locations at that time
    #      : use locations from https://ds.iris.edu/gmap/#network=5A&maxlat=46.5301&maxlon=-120.4604&minlat=46.5217&minlon=-120.4706&drawingmode=box&planet=earth
    #      : and equivalent for UGAP
    stations = project_stations("Rattlesnake Ridge", date)
    # get all station info from IRIS GMAP2 text file
    filepath = "/Users/human/Dropbox/Programs/stingray/projects/rattlesnake_ridge/gmap-stations.txt"
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

def dtm_to_gmt_grid(project_name, UTM=False):
    """
    Takes a DTM file and converts it to a file that is a GMT-compatible grid
    for plotting.

    Example:
    project_name = "Rattlesnake Ridge"
    # write elevation grid in UTM coordinates
    dtm_to_gmt_grid(project_name, UTM=True)
    """

    if project_name == "Rattlesnake Ridge":
        # load arrays from raster file
        raster_file = '/Users/human/Dropbox/Programs/lidar/yakima_basin_2018_dtm_43.tif'

        if UTM:
            # spatial limits
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
                                raster_file, x_limits, y_limits, plot=True,
                                UTM=True)

        # # define header
        # elev_header = [x_limits[0], x_limits[1], y_limits[0], y_limits[1],
        #                x_step, y_step, num_x_steps, num_y_steps]

        # elev_dict['data'] = np.rot90(elevation_grid, k=3)

        # savemat("/Users/human/git/sigpig/sigpig/stingray/srInput"
        #         "/srElevation_TN.mat",
        #         {'srElevation': elev_dict})

        # TODO: convert grid to NetCDF

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

