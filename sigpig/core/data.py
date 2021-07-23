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