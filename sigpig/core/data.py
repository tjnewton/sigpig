"""
Functions to fetch data. 
"""

import obspy
from obspy import read, Stream
from obspy.core.utcdatetime import UTCDateTime
from obspy.clients.fdsn import Client
import calendar


# downloads time series data from IRIS DMC
def get_Waveforms(network, stations, location, channels, start_Time, end_Time):
    """Downloads waveform data using Obspy to access the IRIS DMC
    Find stations via https://ds.iris.edu/gmap/

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

                    st = st.merge(method=1, fill_value=0)
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
                                     nearest_sample=False, fill_value=0)

                        file_st.write("/Users/human/Dropbox/Research/Alaska/build_templates/data/"+filename, format="MSEED")

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

                st = st.merge(method=1, fill_value=0)
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
                                 nearest_sample=False, fill_value=0)

                    file_st.write("/Users/human/Dropbox/Research/Alaska/build_templates/data/" + filename, format="MSEED")

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
    start_Time = UTCDateTime("2018-03-13T00:08:00.0Z")
    end_Time =   UTCDateTime("2018-03-13T00:10:00.0Z")
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


    #FIXME currently limited to same-day queries, use if endtime-starttime > 1 day:

    # FIXME needs to incorporate all channels
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
                           11:19].replace(":", ".")  # use
        # [:19] for date and time
        end_Time_Stamp = str(obspyStream[0].stats.endtime)[11:19].replace(
            ":", ".")
        # writes to snuffler path
        obspyStream.write(f"/Users/human/Dropbox/Programs/snuffler"
                          f"/snuffler_continuous/"
                          f"{project_Aliases[project_Name]}_{start_Time_Stamp}_"
                          f"{end_Time_Stamp}.ms", format="MSEED")

    return obspyStream

