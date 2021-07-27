"""
Functions to identify noise and event onsets in time series.
"""

import obspy
from obspy import read, Stream
from obspy.core.utcdatetime import UTCDateTime
from obspy.clients.fdsn import Client
import calendar

#FIXME: add content from Programs/unet

def picks2array(project_Name: str, marker_File_Path: str, window_Size: int):
    """take in a snuffler marker file and returns a numpy array of window_Size
    long windows with the pick located half way (position window_Size / 2)-1

    Parameters:
        window_Size: int
            length of window in samples

    Example:
        project_Name = "Rattlesnake Ridge"
        marker_File_Path = "picks.mrkr"
        window_Size = 240 # must be divisible by 16 for UNET
        pick_Array, pick_Trace, pick_List = picks2array(project_Name,
        marker_File_Path, window_Size)

        Test out the last result with:
        b = pick_Trace.times()
        pick_time_maybe = pick_Trace.stats.starttime + b[119]
        pick_time_maybe
        pick_time_actually = pick_List[-1]
        pick_time_actually

    # FIXME: check that there is only one pick in the picking window on a
             specific station and channel:  _._ sec
    #    if more than one pick:
    """
    # a place to store the np.arrays
    pick_Array = []

    # function to process a block of picks
    def process_Picks(pick_Array, project_Name, start_time, end_time,  current_Pick_List):
        if project_Name == "Rattlesnake Ridge":
            # load stream with RR data
            channels = ['DP1', 'EHN']
            obspyStream = trim_Daily_Waveforms("Rattlesnake Ridge",
                                               start_time,
                                               end_time, channels,
                                               write_File=False)

        # index station-channel locations in Stream for speed of indexing
        stream_indices = {
            f"{trace.stats.station}.{trace.stats.channel}": index
            for index, trace in enumerate(obspyStream)}

        # loop through picks and assemble np.ndarray of data
        # rows = len(pick_List) # number of picks
        # cols = window_Size # (window_Size / obspyStream[
        # 0].stats.sampling_rate) yields seconds
        # pick_Array = np.empty(shape=(rows, cols))

        for index in current_Pick_List:
            pick = pick_List[index]

            # # is the pick in the current time window?
            # if ((pick[1] + 0.48) > end_time):
            #     continue
            # else:

            # extract window of data centered on the pick,
            # so the pick is at index (window_Size / 2)-1 in the array
            pick_Index = stream_indices[f"{pick[0]}.{pick[2]}"]  # station.channel
            pick_Trace = obspyStream[int(pick_Index)].copy()
            pick_Trace_Starttime = pick[1] - ((window_Size / pick_Trace.stats.sampling_rate) / 2)
            pick_Trace_Endtime = pick[1] + (((window_Size / pick_Trace.stats.sampling_rate) / 2) - pick_Trace.stats.delta)

            pick_Trace.trim(pick_Trace_Starttime, pick_Trace_Endtime)
            # pick_Array[index] = pick_Trace.data
            pick_Array.append(pick_Trace.data)

        print(f"Pick array length: {len(pick_Array)}")

    # build list of picks and dates from the marker file
    pick_List = [] # build a list of tuples of (pick_Station, pick_Time, pick_Index)
    date_List = [] # keep track of dates
    # read the marker file line by line
    with open(marker_File_Path, 'r') as file:
        for line_Contents in file:
            if len(line_Contents) > 52: # avoid irrelevant short lines
                if (line_Contents[0:5] == 'phase') and (line_Contents[-20:-19] == 'P'): # if the line is a P wave pick
                    if len(line_Contents[36:52].split('.')) > 1: # avoids
                        # error from "None" contents
                        #print(line_Contents)
                        pick_Station = line_Contents[36:52].split('.')[1]
                        pick_Channel = line_Contents[36:52].split('.')[3]
                        pick_Channel = pick_Channel.split(' ')[0]
                        pick_Time = UTCDateTime(line_Contents[7:32])
                        pick_List.append((pick_Station, pick_Time,
                                          pick_Channel))

                        # add date to list if it isn't already there
                        if (line_Contents[7:17] not in date_List):
                            date_List.append(line_Contents[7:17])

    # files are for single days, so loop through days and process each day
    for date in date_List:
        # find all items with the date of interest, and start and end times
        day_Pick_List = []
        print(date)

        count = 0
        for index, pick in enumerate(pick_List):
            if (f"{pick[1].year}-{pick[1].month:02}-{pick[1].day:02}" == date):
                day_Pick_List.append(index)

                # find earliest and latest time
                if (count == 0):
                    start_Time = pick[1] - 1 # grabs one sec before pick
                    end_Time = pick[1] + 1
                else:
                    if (start_Time > pick[1] - 1):
                        start_Time = pick[1] - 1
                    if (end_Time < pick[1] + 1):
                        end_Time = pick[1] + 1
                count += 1

        # double check times, may have stepped too far back into
        # previous day or too far forward into next day
        if (f"{start_Time.year}-{start_Time.month:02}-{start_Time.day:02}" !=
                date):
            start_Time = UTCDateTime(f"{date}T00:00:00.000001Z")
        if (f"{end_Time.year}-{end_Time.month:02}-{end_Time.day:02}" !=
                date):
            end_Time = UTCDateTime(f"{date}T23:23:59.99999999Z")

        print(f"Day start time: {start_Time}")
        print(f"Day end time:{end_Time}")

        # if total time is > 1 hour processing may take awhile if the
        # sampling rate between all stations is different. This is due to
        # interpolation on the backend to make all data have the same
        # sampling rate. Thus break processing into 1 hour chunks:
        if ((end_Time - start_Time) > 3600.0):
            # do the splitting and processing
            chunks = int((end_Time - start_Time) / 3600) + 1
            chunk_End_Times = []
            for chunk in range(chunks):
                chunk_End_Times.append(start_Time + (3600.0 * (chunk + 1)))

            print(f"number of chunks: {len(chunk_End_Times)}")

            for chunk_Num, chunk_End_Time in enumerate(chunk_End_Times):
                print(f"Chunk: {chunk_Num + 1}")

                if (chunk_Num == 0):
                    # build chunk pick list
                    chunk_Pick_List = []
                    for index in day_Pick_List:
                        pick = pick_List[index]
                        # if pick occurs on date and time window of interest
                        if (pick[1] > start_Time) and (pick[1] < chunk_End_Time):
                            chunk_Pick_List.append(index)

                    process_Picks(pick_Array, project_Name, start_Time,
                                  chunk_End_Time, chunk_Pick_List)
                    print(f"Chunk start time:{start_Time}")
                    print(f"Chunk end time:{chunk_End_Time}")
                    print(f"Length of chunk pick list: {len(chunk_Pick_List)}")
                else:
                    # build chunk pick list
                    chunk_Pick_List = []
                    for index in day_Pick_List:
                        pick = pick_List[index]
                        # if pick occurs on date and time window of interest
                        if (pick[1] > chunk_End_Times[chunk_Num - 1]) and (
                                pick[1] < chunk_End_Time):
                            chunk_Pick_List.append(index)

                    process_Picks(pick_Array, project_Name, chunk_End_Times[
                        chunk_Num - 1], chunk_End_Time, chunk_Pick_List)
                    print(f"Chunk start time:{chunk_End_Times[chunk_Num - 1]}")
                    print(f"Chunk end time:{chunk_End_Time}")
                    print(f"Length of chunk pick list: {len(chunk_Pick_List)}")

        # < 1 hour case
        else:
            process_Picks(pick_Array, project_Name, start_Time, end_Time, day_Pick_List)
            print(f"Chunk start time:{start_Time}")
            print(f"Chunk end time:{end_Time}")
            print(f"Length of day pick list: {len(day_Pick_List)}")

    # save array as pickle file
    outfile = open('data.pkl', 'wb')
    pickle.dump(pick_Array, outfile)
    outfile.close()

    # return (pick_Array, pick_Trace, pick_List) # just for testing
    return None


# used to be find_Noise_Windows_Nonparallel
def find_Noise_Windows(filepaths: list, detection_Parameters:
tuple, time_Periods: list, filter=False, response=False) -> np.ndarray:
    """
    REVISED (doc not updated) USES FIXED WINDOWS RATHER THAN SLIDING WINDOW.
    ----> see find_Noise_Windows_SlidingWindow for sliding window implementation

    ... Requires files that are Obspy readable, containing data for a single
    station and channel. Different channels of data should be in separate files.
    Uses project_Name for preprocessing steps (use this as a template for
    new projects). A known limitation of this function is that it expects
    all time periods to occur on the same day.

    Parameters:
        filepaths: list of filepaths to mseed files
        detection_Parameters: tuple containing values for the short-term-average window length, the long-term-average window length, the STA/LTA threshold (below this value is considered to be noise), and the desired length of the noise windows in samples. You should experiment to find the optimal parameters for your data.
        time_Periods: list containing tuples of (start_Time, end_Time) for each time period. Multiple time periods are represented as separate list entries, e.g [(start_Time_1, end_Time_1), (start_Time_2, end_Time_2)]. Where start_Time and end_Time are:
            start_Time: UTCDateTime
                defines the start time of the period of interest. Keep in mind
                there is a ramp-up period associated with STA/LTA.
            end_Time: UTCDateTime
                defines the end time of the period of interest

    Example:
        # FIXME: example with actual paths, not my function
        filepaths = project_Filepaths("Rattlesnake Ridge")
        detection_Parameters = (15, 250, 5, 240)                   
        start_Time = UTCDateTime("2018-03-13T01:06:00.0Z")
        end_Time =   UTCDateTime("2018-03-13T01:16:00.0Z")
        time_Periods = [(start_Time, end_Time)]
        noise_Windows, noise_Windows_Times = find_Noise_Windows(filepaths,
        detection_Parameters, time_Periods)
    """

    # set STA/LTA parameters based on unpacked input
    sta, lta, threshold_Value, threshold_Period = detection_Parameters

    # initialize empty list to store noise windows and their times
    noise_Windows = []
    noise_Windows_Times = []

    # loop through filepaths
    for filepath in filepaths:
        # time the run
        start = time.time()
        # define the station name and channel from filepath
        # EDIT THIS SECTION TO MATCH YOUR FILEPATH FORMAT
        # my filepaths look like the following:
        # /Users/human/Desktop/RR_MSEED/5A.27..DP1.2018-03-13T00.00.00.ms
        # so I extract the station from the filename.
        # First, isolate the filename from the filepath (assuming filename
        # doesn't contain /)
        filename = filepath.split("/")[-1]
        # then grab the station name from the filename
        station = filename.split(".")[1]
        channel = filename.split(".")[3]
        print(f" Processing station: {station}, channel: {channel}")
        # initialize empty list to store noise windows unique to this station and channel
        station_Noise_Windows = []
        station_Noise_Windows_Times = []

        # loads data, merges traces if there is more than one, selects first trace
        trace = read(filepath).merge(method=1, fill_value=0)[0]

        # loop through time periods, ASSUMING they all occur on the same day
        for start_Time, end_Time in time_Periods:

            # copy trace and trim to specified time period
            period_Trace = trace.copy()
            period_Trace = period_Trace.trim(start_Time, end_Time)

            # correct for instrument response?
            if response:
                if period_Trace.stats.network == "UW":
                    invfile = \
                        '/Users/human/git/noisepicker-private/resps/RESP.' + period_Trace.stats.station \
                        + '.' + period_Trace.stats.network + '..' + period_Trace.stats.channel
                else:
                    invfile = '/Users/human/git/noisepicker-private/resps/RESP.' + period_Trace.stats.network \
                              + '.' + period_Trace.stats.station + '..' + period_Trace.stats.channel
                inv = read_inventory(path_or_file_object=invfile,
                                     format='RESP')
                period_Trace.remove_response(inventory=inv, output='VEL',
                                             plot=False)

            # bandpass filter and detrend data?
            if filter:
                period_Trace.filter("bandpass", freqmin=20, freqmax=60,
                                    corners=4)
                period_Trace.detrend('linear')

            times = period_Trace.times("utcdatetime")
            data = period_Trace.data

            # STA/LTA via Obspy
            cft = classic_sta_lta(data, sta, lta)

            # find where STA/LTA < threshold
            stalta_Mask = np.where(cft < threshold_Value, True, False)

            # exclude edge effects from mask (where STA/LTA is zero at start and end of timeseries)
            for index in range(
                    len(
                        cft)):  # counts up to capture beginning edge effect range
                if cft[index] != 0:
                    stalta_Mask[0: index] = False
                    break
            # # not necessary for standard STA/LTA
            # for index in range(len(cft)-1, 0, -1): # counts down to capture ending edge effect range
            #     if cft[index] != 0:
            #         stalta_mask[index:len(cft)] = False
            #         break

            # loop through each time step (sample) and find noise windows of
            # the specified length in samples
            for index in range(0, len(stalta_Mask) - threshold_Period,
                               threshold_Period):
                # print(f"Station: {station}  Channel: {channel}  : : :"
                #       f" {index} of {len(stalta_Mask)}")
                if stalta_Mask[
                    index]:  # if value of cft at index is < STA/LTA threshold
                    if all(stalta_Mask[index: (
                            index + threshold_Period)]):  # if all values in the specified window are < STA/LTA threshold
                        # extract start and end time for window
                        noise_Start = times[index]
                        noise_End = times[index + threshold_Period]

                        # append noise window and metadata to lists
                        station_Noise_Windows.append(data[index: (
                                index + threshold_Period)])  # append data within noise window
                        station_Noise_Windows_Times.append((noise_Start,
                                                            noise_End, station,
                                                            channel))

            # append noise windows one at a time to major list
            if len(station_Noise_Windows) == 0:
                print(f"{start_Time}----{end_Time}      Found 0 noise "
                      f"windows.")
                pass  # list is empty, get over it :(
            else:
                print(f"{start_Time}----{end_Time}      Found"
                      f" {len(station_Noise_Windows)} noise windows.")
                for window_index in range(len(station_Noise_Windows)):
                    noise_Windows.append(station_Noise_Windows[window_index])
                    noise_Windows_Times.append(
                        station_Noise_Windows_Times[window_index])

            # print the time to process this station/channel
            end = time.time()
            print(f"station/channel runtime: {(end - start) / 60:.2f} "
                  f"minutes.")

    # convert to np.ndarray
    noise_Windows = np.array(noise_Windows)

    # save array as pickle file
    outfile = open('noise_windows.pkl', 'wb')
    pickle.dump(noise_Windows, outfile)
    outfile.close()

    outfile = open('noise_times.pkl', 'wb')
    pickle.dump(noise_Windows_Times, outfile)
    outfile.close()

    return (noise_Windows, noise_Windows_Times)


def plot_Noise_Windows(noise_Windows, noise_Windows_Times=False,
                       indices=False):
    """Plots normalized noise windows from find_Noise_Windows function,
    optionally plots data with timestamps. Plot without timestamps in more
    compact. If plotting without timestamps, the indices can be plotted with
    the noise windows by supplying a numpy array containing the indices.


    # FIXME this function breaks when plotting > 172 windows. I think
    # this has to do with matplotlib but I have no proof for this claim
    # and I have not investigated further because life. If you need to
    # plot >172 noise windows at a time, good luck on your journey.

    Examples
    --------
    import pickle
    # load run details for program runtime and number of noise windows
    infile = open('run_details.pkl','rb')
    run_Details = pickle.load(infile)
    infile.close()
    print(f"Runtime: {run_Details[0] / 60 / 60} hours.")
    print(f"Found {run_Details[1]} noise windows.")
    # load noise windows from pickle file
    infile = open('noise_windows.pkl', 'rb')
    noise_Windows = pickle.load(infile)
    infile.close()
    # load info associated with noise windows (times, station, channel)
    infile = open('noise_times.pkl', 'rb')
    noise_Windows_Times = pickle.load(infile)
    infile.close()

    # plot 'em utilizing slicing
    plot_Noise_Windows(noise_Windows[:172]) # breaks when plotting > 172 lines
    # or plot 'em with their associated times
    plot_Noise_Windows(noise_Windows[:20], noise_Windows_Times[:20])
    """

    # normalize by max value in each window to
    maxTraceValue = max_Amplitude(noise_Windows)

    # if window times are provided, plot them. This is not a very compact
    # figure, hence the option to plot without times.
    if noise_Windows_Times:
        # set the figure size
        figureWidth = 19
        figureHeight = 1.5 * len(noise_Windows)

        fig, ax = plt.subplots(nrows=len(noise_Windows), sharex=False,
                               figsize=(figureWidth, figureHeight))
        for index in range(len(noise_Windows)):
            # build list of times for each window based on start and end times
            ax[index].plot_date(noise_Windows_Times[index][2], noise_Windows[
                index] / maxTraceValue[index], fmt="k-", linewidth=0.4)
            xlabels = [noise_Windows_Times[index][0].matplotlib_date,
                       noise_Windows_Times[index][1].matplotlib_date]
            ax[index].set_xticks(xlabels)
            # format x labels
            myFmt = DateFormatter("%H:%M:%S.%f")
            ax[index].xaxis.set_major_formatter(myFmt)
            ax[index].set_xlim(xlabels)

        plt.xlabel("Time")
        plt.show()

    else:
        # set the figure size
        figureWidth = 19
        figureHeight = 0.8 * len(noise_Windows)

        # trim number of noise windows if > 172
        if len(noise_Windows) > 172:
            noise_Windows = noise_Windows[0:171]

        # if no indices are specified, set them to arbitrary numbers
        if type(indices) == bool:
            indices = list(range(0, len(noise_Windows)))

        fig, ax = plt.subplots(nrows=1, sharex=True,
                               figsize=(figureWidth, figureHeight))
        for index in range(len(noise_Windows)):
            ax.plot(range(len(noise_Windows[index])), (noise_Windows[
                                                           index] /
                                                       maxTraceValue[
                                                           index]) * 0.7 + index + min(
                indices), "k-", linewidth=0.4)

        ax.set_xlim([0, len(noise_Windows[0])])
        ax.set_ylim([min(indices) - 1, max(indices) + 1])  # [0,
        # len(noise_Windows) + 1]
        plt.ylabel("Window #")
        plt.xlabel("Sample")
        plt.yticks(indices)
        plt.show()


def explore_Noise_Windows(noise_Windows, noise_Windows_Times,
                          priors_Pickle=False):
    """Plots normalized noise windows (from find_Noise_Windows function)
    in a manner that allows visual verification. Utilizes user input to
    record false-noise-windows during visualization and returns two objects
    containing noise windows and noise windows times (similar to noise_Windows
    and noise_Windows_Times) with false noise windows and their corresponding
    time data removed, upon reaching (and plotting) the last noise window in
    noise_Windows. To reiterate, this function only returns the "verified"
    noise windows and times upon reaching the end of the noise windows. To use
    this function multiple times on the same noise_Windows and
    noise_Windows_Times objects, supply the function with name of pickle file
    as priors_Pickle to specify noise window indices to remove upon reaching
    the last noise window. Type quit to quit, or type nothing to plot the next
    batch of noise windows.

    Parameters
    ----------
    noise_Windows: noise windows as returned from find_Noise_Windows function
    noise_Windows_Times: noise windows times as returned from find_Noise_Windows function
    priors_Pickle: pickle file containing list. First item of list contains
                start_Index, second item of list (index 1) is a list of
                indices of noise windows flagged for removal. Where
                start_Index is...
                start_Index: int, index of noise window to start plotting.
                            Function will plot from start_Index to the end
                            of the noise windows in increments of 172 (the
                            max that plot_Noise_Windows can plot at one time).

    Examples
    --------
    # import the relevant function
    from noisepicker import explore_Noise_Windows

    # load noise windows from pickle file
    infile = open(f'noise_windows_for_validation.pkl', 'rb')
    noise_Windows = pickle.load(infile)
    infile.close()

    print(f"Noise windows before validation: {len(noise_Windows)}")

    # call explore_Noise_Windows to begin visual validation of noise windows
    validated_Noise_Windows = explore_Noise_Windows(noise_Windows)

    if type(validated_Noise_Windows) == np.ndarray:
        print(f"Noise windows after validation: {len(validated_Noise_Windows)}")
        # then save the returned noise windows once you reach the end of the batch
        outfile = open('validated_noise_windows.pkl', 'wb')
        pickle.dump(validated_Noise_Windows, outfile)
        outfile.close()

    """

    # function to save progress to pickle file
    def save_Progress(index, removal_Indices):
        priors = []
        priors.append(index)
        priors.append(removal_Indices)

        # save list as pickle file
        outfile = open('priors_Pickle.pkl', 'wb')
        pickle.dump(priors, outfile)
        outfile.close()

    # function to remove specified indices from noise_Windows
    def remove_Windows(noise_Windows, noise_Windows_Times, removal_Indices):
        # sort indices to avoid unexpected behavior
        removal_Indices.sort()
        # then remove each index from noise windows
        verified_Noise_Windows = np.delete(noise_Windows, removal_Indices,
                                           axis=0)
        verified_Noise_Windows_Times = np.delete(noise_Windows_Times,
                                                 removal_Indices,
                                                 axis=0)
        return verified_Noise_Windows, verified_Noise_Windows_Times

    # set verified_Noise_Windows to None. Only returns noise windows with
    # false noise windows removed once end of noise windows is reached.
    verified_Noise_Windows = None
    verified_Noise_Windows_Times = None

    # check for pickle file with prior validation info
    if type(priors_Pickle) != bool:
        # load the pickle, set the start_Index, save the indices
        infile = open(priors_Pickle, 'rb')
        priors = pickle.load(infile)
        infile.close()
        start_Index = priors[0]
        removal_Indices = priors[1]
    else:
        start_Index = 0
        removal_Indices = []

    # create a flag to detect the end of noise_Windows
    endFlag = False

    # loop through and plot 172 noise windows at a time
    for index in range(start_Index, len(noise_Windows), 172):
        # set the number of windows to plot, defaults to 172 unless there
        # are less than 172 noise windows left
        if (len(noise_Windows) - index > 172):
            num_Windows = 172
        else:
            num_Windows = len(noise_Windows) - index

        # put the endFlag up if warranted
        if num_Windows != 172:
            endFlag = True

        # plot the noise windows and their indices
        plot_Noise_Windows(noise_Windows[index: index + num_Windows],
                           indices=list(range(index, index + num_Windows)))

        # ask for user input: space separated indices of noise windows to
        # flag for removal, or quit, or nothing
        user_Input = input("Enter 'quit', nothing, or space separated indices "
                           "of noise windows to be flagged for removal: ")

        # if input is "quit", quit.
        if (user_Input == "quit") or (user_Input == "Quit"):
            # save progress and start at same index next time
            save_Progress(index, removal_Indices)
            break

        # if there is no input, go to next plot (if there is a next plot)
        elif (user_Input == ""):
            # do nothing but continue unless the end has been reached
            if endFlag:
                # set saved index to last item in noise_Windows
                save_Progress(len(noise_Windows) - 1, removal_Indices)
                if len(removal_Indices) > 0:
                    verified_Noise_Windows, verified_Noise_Windows_Times = remove_Windows(
                        noise_Windows, noise_Windows_Times,
                        removal_Indices)

        # case for user input that isn't blank or quit, process the input
        else:
            # split string at spaces
            split_Input = user_Input.split()

            # append items to removal_Indices
            if len(split_Input) > 0:
                for item in split_Input:
                    removal_Indices.append(int(item))

            if endFlag:
                # set saved index to last item in noise_Windows
                save_Progress(len(noise_Windows) - 1, removal_Indices)
                if len(removal_Indices) > 0:
                    verified_Noise_Windows, verified_Noise_Windows_Times = remove_Windows(
                        noise_Windows, noise_Windows_Times,
                        removal_Indices)

    return verified_Noise_Windows, verified_Noise_Windows_Times


def verify_Noise_Windows(noise_Windows: np.ndarray, noise_Windows_Times:
list, marker_File_Path: str):
    """Removes false-positive noise windows from a pickle file (output by
    find_Noise_Windows function) by identifying "noise windows" that contain
    first arrival picks. This means you should pick all first arrivals for
    the time period in which you are searching for noise windows.

    Parameters:

    Example:
        import pickle
        # load run details for program runtime and number of noise windows
        infile = open('0_run_details.pkl','rb')
        run_Details = pickle.load(infile)
        infile.close()
        print(f"Runtime: {run_Details[0] / 60 / 60} hours.")
        print(f"Found {run_Details[1]} noise windows.")
        # load noise windows from pickle file
        infile = open('0_noise_windows.pkl', 'rb')
        noise_Windows = pickle.load(infile)
        infile.close()
        # load info associated with noise windows (times, station, channel)
        infile = open('0_noise_times.pkl', 'rb')
        noise_Windows_Times = pickle.load(infile)
        infile.close()

        marker_File_Path = 'picks.mrkr'
        verified_Noise_Windows, verified_Noise_Times = verify_Noise_Windows(noise_Windows, noise_Windows_Times, marker_File_Path)

    """
    pick_List = []  # build a list of tuples of (pick_Station, pick_Time, pick_Index)

    # # for testing
    # file = open("picks.mrkr", "r")
    # print(file.readline())
    # file.close()

    # read the marker file line by line
    with open(marker_File_Path, 'r') as file:
        for line_Contents in file:
            if len(line_Contents) > 52:  # avoid irrelevant short lines
                if (line_Contents[0:5] == 'phase') and (line_Contents[
                                                        -20:-19] == 'P'):  # if the line is a P wave pick
                    if len(line_Contents[36:52].split(
                            '.')) > 1:  # avoids error from "None" contents
                        pick_Station = line_Contents[36:52].split('.')[1]
                        pick_Channel = line_Contents[36:52].split('.')[3]
                        pick_Channel = pick_Channel.split(' ')[0]
                        pick_Time = UTCDateTime(line_Contents[7:32])
                        pick_List.append((pick_Station, pick_Time,
                                          pick_Channel))

    # check for picks within noise windows and build list of indices to remove
    false_Noise_Indices = []
    for index, window in enumerate(noise_Windows_Times):
        print(f"Processing window {index} of {len(noise_Windows_Times)}")
        false_Window_Flag = False
        for pick in pick_List:
            # check that station and channel are same for the pick and the
            # window, then check that pick is within window time bounds
            station_Flag = pick[0] == window[2]
            channel_Flag = pick[2] == window[3]
            time_Flag = (pick[1] >= window[0]) and (pick[1] <= window[1])
            if station_Flag and channel_Flag and time_Flag:
                false_Window_Flag = True
                break
        if false_Window_Flag:
            false_Noise_Indices.append(index)

    # remove false noise windows from list of indices
    verified_Noise_Windows = np.delete(noise_Windows, false_Noise_Indices,
                                       axis=0)
    # make a copy of the list as to not modify in place
    verified_Noise_Windows_Times = copy.deepcopy(noise_Windows_Times)
    # then remove the indices from the list of times
    for index in sorted(false_Noise_Indices, reverse=True):
        del verified_Noise_Windows_Times[index]

    print(f"Found {len(false_Noise_Indices)} false noise windows containing "
          f"picks.")

    return (verified_Noise_Windows, verified_Noise_Windows_Times)


def plot_Noise_Time_Histogram(noise_Windows_Times: list, bin_Width:
int, major_X_Tick="day", minor_X_Tick="hour"):
    """
    Plots histogram of noise windows to visualize their temporal distribution.
    Bin width must be specified in seconds. 

    Example:

    """

    # first initialize start_Time and end_Time to find temporal bounds of data
    start_Time = noise_Windows_Times[0][0]
    end_Time = noise_Windows_Times[0][1]

    # find temporal bounds of dataset by looping through each item
    for item in noise_Windows_Times:
        if item[0] < start_Time:
            start_Time = item[0]
        if item[1] > end_Time:
            end_Time = item[1]

    # build data bins based on start and end times and specified bin width
    number_Of_Bins = int((end_Time - start_Time) / bin_Width) + 1
    bins = [start_Time + (index * bin_Width) for index in
            range(number_Of_Bins)]

    # determine number of noise windows per bin
    # FIXME: this is slow. speed it up.
    print("Finding the counts per bin is slow...")
    counts = []
    total = len(bins)
    for index, bin in enumerate(bins):
        print(f"{index + 1} of {total}")
        count = 0
        # check if start of noise window is within bin, if so add to count
        for item in noise_Windows_Times:
            if (item[0] >= bin) and (item[0] < (bin + bin_Width)):
                count += 1
        counts.append(count)

    plt.style.use('Solarize_Light2')
    fig = plt.figure(figsize=(20, 4), dpi=150)
    ax = fig.add_subplot(111)
    # add space for a second x-axis (not sure what a good value for bottom is)
    fig.subplots_adjust(bottom=0.2)

    # ax.bar(days, values, align='edge', width=1, linewidth=0, alpha=.8)
    # convert bins to Python datetime objects
    bins = [item.datetime for item in bins]
    ax.bar(bins, counts, align='edge', width=1 / (len(bins) + 1),
           linewidth=1.0,
           edgecolor="black", alpha=.8)
    ax2 = ax.twiny()

    # define different major tick conventions for different time spans
    MAJOR_TICK_LOCATOR = {"month": dates.MonthLocator(), "hour":
        dates.HourLocator(byhour=[0, 6, 12, 18, 24]), "day":
                              dates.DayLocator(
                                  bymonthday=[start_Time.datetime.day])}
    MAJOR_TICK_FORMATTER = {"month": dates.DateFormatter('%b'), "hour":
        dates.DateFormatter('%-H'), "day":
                                dates.DateFormatter('%b %-d %Y')}

    # X labels are split between two axes for spacing and readability
    # set axis labels, axis 2: major axis
    ax2.xaxis.set_major_locator(MAJOR_TICK_LOCATOR[major_X_Tick])
    ax2.xaxis.set_major_formatter(MAJOR_TICK_FORMATTER[major_X_Tick])

    # define different minor tick conventions for different time spans
    MINOR_TICK_LOCATOR = {"day": dates.DayLocator(bymonthday=[1, 8, 15, 22]),
                          "minute": dates.MinuteLocator(byminute=[0, 30]),
                          "hour": dates.HourLocator(byhour=[index for index
                                                            in range(24)])}
    MINOR_TICK_FORMATTER = {"day": dates.DateFormatter('%-d'), "minute":
        dates.DateFormatter('%-M'), "hour": dates.DateFormatter('%-H')}

    # axis 1: minor axis (despite setting the major axis here)
    ax.xaxis.set_major_locator(MINOR_TICK_LOCATOR[minor_X_Tick])
    ax.xaxis.set_major_formatter(MINOR_TICK_FORMATTER[minor_X_Tick])
    ax.xaxis.set_tick_params(labelsize=8)

    # define xlim dict for dynamic time scaling, round down start time
    # and up end time to make the plot look tidy
    X_LIMITS = {"hour": {"start_Time": start_Time.datetime.replace(
        microsecond=0, second=0, minute=0), "end_Time":
        end_Time.datetime.replace(microsecond=999999, second=59,
                                  minute=59, hour=23,
                                  day=start_Time.datetime.day)}}
    ax.set_xlim(X_LIMITS[minor_X_Tick]["start_Time"], X_LIMITS[
        minor_X_Tick]["end_Time"])

    # # Axis 1: Minor (Days 8 / 22)
    # ax.xaxis.set_minor_locator(dates.DayLocator(bymonthday=[8, 22]))
    # ax.xaxis.set_minor_formatter(dates.DateFormatter('%-d'))
    # ax.xaxis.set_tick_params(which="minor", labelsize=5)

    # This is a lot of work just to move the second axis to the bottom
    # of the chart (we made room for it with subplots_adjust above)
    ax2.set_xlim(ax.get_xlim())
    ax2.spines["bottom"].set_position(("axes", -0.05))  # position of text.
    ax2.spines["bottom"].set_visible(False)  # don't show the axis line
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.xaxis.set_tick_params(grid_visible=False)  # already have ax's grid
    for label in ax2.xaxis.get_ticklabels():
        label.set_horizontalalignment('left')
    # ax2.patch.set_visible(False)
    # for sp in ax2.spines.values(): sp.set_visible(False)

    ax.set_title("Temporal Histogram of Noise")
    ax.set_title(
        "{:,} noise windows".format(len(noise_Windows_Times)),
        loc='right', fontsize=8, color='gray')
    ax2.set_xlabel("Hour of Day")
    ax.set_ylabel("Count")

    # save the figure
    # fig.savefig(
    #     'plot_by_day.png',
    #     facecolor=fig.get_facecolor(),
    #     edgecolor='none'
    # )

    # plot the figure
    plt.show()
    return None


def plot_Noise_Station_Histogram(noise_Windows_Times: list, stations: list):
    """
    Plots histogram of noise windows to visualize their spatial distribution.
    """
    # build data bin counts based on stations
    # FIXME: this is slow. speed it up.
    print("Finding the counts per bin is slow...")
    counts = []
    total = len(stations)
    for index, bin in enumerate(stations):
        print(f"{index + 1} of {total}")
        count = 0
        # check if start of noise window is within bin, if so add to count
        for item in noise_Windows_Times:
            if item[2] == bin:
                count += 1
        counts.append(count)

    plt.style.use('Solarize_Light2')
    fig = plt.figure(figsize=(20, 4), dpi=150)
    ax = fig.add_subplot(111)
    # add space for a second x-axis (not sure what a good value for bottom is)
    fig.subplots_adjust(bottom=0.2)

    # ax.bar(days, values, align='edge', width=1, linewidth=0, alpha=.8)
    # convert station bins to x locations
    x_Coordinates = [index for index in range(len(stations))]
    ax.bar(x_Coordinates, counts, align='center', width=1,
           linewidth=1.0, edgecolor="black", alpha=.8)
    ax2 = ax.twiny()

    # define different major tick conventions for different time spans
    MAJOR_TICK_LOCATOR = {"day": dates.DayLocator(
        bymonthday=[noise_Windows_Times[0][0].datetime.day])}
    MAJOR_TICK_FORMATTER = {"day": dates.DateFormatter('%b %-d %Y')}

    # X labels are split between two axes for spacing and readability
    # set axis labels, axis 2: major axis
    ax2.xaxis.set_major_locator(MAJOR_TICK_LOCATOR["day"])
    ax2.xaxis.set_major_formatter(MAJOR_TICK_FORMATTER["day"])

    # define different minor tick conventions for stations
    ax.xaxis.set_major_locator(plt.FixedLocator(x_Coordinates))
    ax.xaxis.set_major_formatter(plt.FixedFormatter(stations))
    ax.xaxis.set_tick_params(labelsize=8)

    ax.set_xlim(-0.6, x_Coordinates[-1] + 0.6)

    # # Axis 1: Minor (Days 8 / 22)
    # ax.xaxis.set_minor_locator(dates.DayLocator(bymonthday=[8, 22]))
    # ax.xaxis.set_minor_formatter(dates.DateFormatter('%-d'))
    # ax.xaxis.set_tick_params(which="minor", labelsize=5)

    # # This is a lot of work just to move the second axis to the bottom
    # # of the chart (we made room for it with subplots_adjust above)
    # ax2.set_xlim(ax.get_xlim())
    # ax2.spines["bottom"].set_position(("axes", -0.05))  # position of text.
    # ax2.spines["bottom"].set_visible(False)  # don't show the axis line
    # ax2.xaxis.set_ticks_position("bottom")
    # ax2.xaxis.set_label_position("bottom")
    # ax2.xaxis.set_tick_params(grid_visible=False)  # already have ax's grid
    # for label in ax2.xaxis.get_ticklabels():
    #     label.set_horizontalalignment('left')
    # # ax2.patch.set_visible(False)
    # # for sp in ax2.spines.values(): sp.set_visible(False)

    ax.set_title("Histogram of Noise by Station")
    ax.set_title(
        "{:,} noise windows".format(len(noise_Windows_Times)),
        loc='right', fontsize=8, color='gray')
    ax.set_title(f"{noise_Windows_Times[0][0].datetime.strftime('%b %-d %Y')}",
                 loc='left', fontsize=8, color='gray')
    ax.set_xlabel("Station")
    ax.set_ylabel("Count")

    # save the figure
    # fig.savefig(
    #     'plot_by_day.png',
    #     facecolor=fig.get_facecolor(),
    #     edgecolor='none'
    # )

    # plot the figure
    plt.show()
    return None


def analyze_Noise_Windows(noise_Windows: np.ndarray,
                          noise_Windows_Times: list):
    """Calculates metrics (mean of absolute value of window, and median of
    absolute value of window) based the contents of each noise window and
    plots the distribution of these metrics accross stations.

    Parameters:

    Example:


    """
    metric_Map = {}
    # min_X = 0
    # max_X = 0
    # min_Y = 0
    # max_Y = 0

    # gather metrics for all noise windows
    for index, item in enumerate(noise_Windows_Times):
        # store max(abs()) and median(abs)
        metric_Tuple = (np.max(np.abs(noise_Windows[index])),
                        np.median(np.abs(noise_Windows[index])),
                        np.sqrt(np.mean(np.square(noise_Windows[index]))))
        if item[2] in metric_Map:
            metric_Map[item[2]].append(metric_Tuple)
        else:
            metric_Map[item[2]] = [metric_Tuple]

    # plot metrics in ridgeplots
    stations = ['1', '2', '3', '5', '4', '6', '7', '8', '13', '9', '10',
                '12', '15', 'UGAP3', '16', '17', '18', '20', '21', '22',
                '23', '25', '26', '27', 'UGAP5', 'UGAP6', '28', '30', '31',
                '32', '33', '34', '35', '36', '37', '38', '39', '40', '41',
                '42']
    cmap = mpl.cm.get_cmap("magma", len(stations))
    colors = cmap.colors
    # colors = ['#0000ff', '#3300cc', '#660099', '#990066', '#cc0033', '#ff0000'] # make longer

    gs = grid_spec.GridSpec(len(stations), 1)
    fig = plt.figure(figsize=(14, 30))

    # build file with station, mean, median
    save_Metrics = []

    ax_objs = []

    x_min = -0.5
    x_max = 0.5
    y_max = 20.0
    for i, station in enumerate(stations):
        # index 1 in metric Map is median, 0 is max
        median_Metrics = [metric_Map[station][i][1] for i in
                          range(len(metric_Map[station]))]
        median_Metrics = np.array(median_Metrics)

        rms_Metrics = [metric_Map[station][i][2] for i in
                       range(len(metric_Map[station]))]
        rms_Metrics = np.array(rms_Metrics)
        # x_d = np.linspace(x_min, x_max, 6000)

        # kde = KernelDensity(bandwidth=0.03, kernel='gaussian')
        # kde.fit(x[:, None])

        # logprob = kde.score_samples(x_d[:, None])

        # creating new axes object
        # ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))

        # plotting the distribution
        # ax_objs[-1].plot(x_d, np.exp(logprob),color="red",lw=1)
        # color=colors[i] above
        # ax_objs[-1].fill_between(x_d, np.exp(logprob), alpha=1, color="red")

        # plot mean and median
        # y_points = [0, 0.05]
        # means = [np.mean(x), np.mean(x)]
        # medians = [np.median(x), np.median(x)]
        # ax_objs[-1].plot(means, y_points, color="green", lw=1)
        # ax_objs[-1].plot(medians, y_points, color="blue", lw=1)

        # save the relevant info
        # rms = np.sqrt(np.mean(np.square(x)))
        save_Metrics.append([station, str(np.mean(median_Metrics)),
                             str(np.median(median_Metrics)),
                             str(np.mean(rms_Metrics)),
                             str(np.median(rms_Metrics))])

        # setting uniform x and y lims
        # ax_objs[-1].set_xlim(x_min, x_max)
        # ax_objs[-1].set_ylim(0, y_max)

        # make background transparent
        # rect = ax_objs[-1].patch
        # rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        # ax_objs[-1].set_yticklabels([])
        #
        # if i == len(stations)-1:
        #     ax_objs[-1].set_xlabel("Median of Absolute Value", fontsize=16,fontweight="bold")
        # else:
        #     ax_objs[-1].set_xticklabels([])
        #
        # spines = ["top","right","left","bottom"]
        # for s in spines:
        #     ax_objs[-1].spines[s].set_visible(False)
        #
        # adj_station = station.replace(" ","\n")
        # ax_objs[-1].text(x_min,0,adj_station,fontweight="bold",fontsize=14,
        #                  ha="right")

    # gs.update(hspace=-0.7)

    # fig.text(0.13,0.85,f"Distribution of Median(Absolute_Value(Noise_"
    #                    f"Window)) n={len(noise_Windows)}",fontsize=20)

    # plt.tight_layout()
    # plt.savefig('median_SNR_filtered.png', dpi=300)
    # plt.show()

    # save as csv file
    save_Metrics = np.array(save_Metrics)
    np.savetxt("noise_metrics.csv", save_Metrics, delimiter=",", fmt='%s')
    print(save_Metrics)

    # # max plot
    # fig = plt.figure(figsize=(14,30))
    # ax_objs = []
    # for i, station in enumerate(stations):
    #     # index 1 in metric Map is median, 0 is max
    #     median_Metrics = [metric_Map[station][i][0] for i in range(len(
    #         metric_Map[station]))]
    #     x = np.array(median_Metrics)
    #     x_d = np.linspace(x_min, x_max, 6000)
    #
    #     kde = KernelDensity(bandwidth=0.03, kernel='gaussian')
    #     kde.fit(x[:, None])
    #
    #     logprob = kde.score_samples(x_d[:, None])
    #
    #     # creating new axes object
    #     ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))
    #
    #     # plotting the distribution
    #     ax_objs[-1].plot(x_d, np.exp(logprob),color="red",lw=1)
    #     #color=colors[i] above
    #     ax_objs[-1].fill_between(x_d, np.exp(logprob), alpha=1,
    #                              color="red")
    #
    #
    #     # setting uniform x and y lims
    #     ax_objs[-1].set_xlim(x_min, x_max)
    #     ax_objs[-1].set_ylim(0, y_max)
    #
    #     # make background transparent
    #     rect = ax_objs[-1].patch
    #     rect.set_alpha(0)
    #
    #     # remove borders, axis ticks, and labels
    #     ax_objs[-1].set_yticklabels([])
    #
    #     if i == len(stations)-1:
    #         ax_objs[-1].set_xlabel("Max of Absolute Value", fontsize=16,
    #                                fontweight="bold")
    #     else:
    #         ax_objs[-1].set_xticklabels([])
    #
    #     spines = ["top","right","left","bottom"]
    #     for s in spines:
    #         ax_objs[-1].spines[s].set_visible(False)
    #
    #     adj_station = station.replace(" ","\n")
    #     ax_objs[-1].text(x_min,0,adj_station,fontweight="bold",fontsize=14,
    #                      ha="right") # x_min was -0.02
    #
    # gs.update(hspace=-0.7)
    #
    # fig.text(0.13,0.85,f"Distribution of Max(Absolute_Value(Noise_"
    #                    f"Window)) n={len(noise_Windows)}",fontsize=20)
    #
    # #plt.tight_layout()
    # plt.savefig('max_SNR_filtered.png', dpi=300)
    # plt.show()

    return metric_Map




