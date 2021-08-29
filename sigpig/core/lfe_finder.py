"""
Functions to find low-frequency earthquakes in seismic time series data.
"""

import logging
import obspy
from obspy import UTCDateTime, Stream, Trace, read, read_events
from obspy.signal.cross_correlation import xcorr, correlate_template
from eqcorrscan import Tribe, Party, Family, Template
from eqcorrscan.utils.clustering import cluster
from eqcorrscan.utils.stacking import PWS_stack, linstack, align_traces
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import calendar
from tqdm import tqdm
from figures import plot_stack
from scipy.signal import hilbert
import time
from data import max_amplitude

# function to convert snuffler marker file to event template
def markers_to_template(marker_file_path, prepick_offset, time_markers=False):
    """
    Loads snuffler marker file (for a single event) and generates template
    objects required for signal detection via matched-filter analysis with
    EQcorrscan.

    Limitations: built to work with a single event.

    Example:
        # define the path to the snuffler marker file
        marker_file_path = "lfe_template.mrkr"

        # define the offset from the S wave pick to start of template
        prepick_offset = 11 # in seconds

        templates, station_dict, pick_offset = markers_to_template(marker_file_path, prepick_offset)

        # print each line in templates without newline character
        for line in templates:
            print(line[:-1])

        # print contents of station dict
        for station in station_dict.keys():
            print(f"{station} {station_dict[station]}")

        # print contents of pick offset dict
        for station in pick_offset.keys():
            print(f"{station} {pick_offset[station]}")

    """

    # build dict of picks from the marker file
    pick_dict = {}
    station_dict = {}
    template_windows = {}
    pick_offset = {}
    # keep track of earliest pick time
    earliest_pick_time = []

    # read the marker file line by line
    with open(marker_file_path, 'r') as file:
        for line_Contents in file:
            if len(line_Contents) > 52:  # avoid irrelevant short lines
                if (line_Contents[0:5] == 'phase') and (line_Contents[
                                                        -20:-19] == 'S'):
                    # avoids error from "None" contents
                    if len(line_Contents[35:51].split('.')) > 1:
                        # print(line_Contents[7:31])
                        pick_station = line_Contents[35:51].split('.')[1]
                        pick_channel = line_Contents[35:51].split('.')[3]
                        pick_channel = pick_channel.split(' ')[0]
                        pick_channel = pick_channel[:-1] + "Z"
                        pick_network = line_Contents[35:51].split('.')[0]
                        pick_time = UTCDateTime(line_Contents[7:31])
                        pick_dict[pick_station] = pick_time

                        # print(f"{pick_network} {pick_station} "
                        #       f"{pick_channel} {pick_time}")

                        # build station_dict
                        station_dict[pick_station] = {"network": pick_network,
                                                      "channel": pick_channel}

                        # build template window object for plotting, 16 s templ
                        template_windows[f"{pick_network}.{pick_station}"] =\
                            [pick_time - prepick_offset, (pick_time -
                                                          prepick_offset) + 16]

                        # check if this is the earliest pick time
                        if len(earliest_pick_time) == 0:
                            earliest_pick_time.append(pick_time)
                            earliest_pick_station = pick_station
                        else:
                            # redefine earliest if necessary
                            if pick_time < earliest_pick_time[0]:
                                earliest_pick_time[0] = pick_time
                                earliest_pick_station = pick_station

    # account for prepick offset in earliest pick
    earliest_pick_time = earliest_pick_time[0] - prepick_offset

    # build templates object header (location is made up)
    templates = [f"# {earliest_pick_time.year} {earliest_pick_time.month:02} "
                 f"{earliest_pick_time.day:02} {earliest_pick_time.hour:02} "
                 f"{earliest_pick_time.minute:02} "
                 f"{earliest_pick_time.second:02}."
                 f"{earliest_pick_time.microsecond:02}  61.8000 "
                 f"-144.0000  30.00  1.00  0.0  0.0  0.00  1\n"]

    # append earliest station to templates list
    templates.append(f"{earliest_pick_station}    0.000  1       P\n")

    # create pick offset dict for pick offset from main trace
    pick_offset[earliest_pick_station] = 0.0

    # append all other stations to templates list
    for station in pick_dict.keys():
        if station != earliest_pick_station:
            time_diff = pick_dict[station] - (earliest_pick_time + prepick_offset)
            microseconds = int((time_diff - int(time_diff)) * 1000)
            # print(f"time_diff: {int(time_diff)} microseconds:{microseconds}")
            templates.append(f"{station:4}   {int(time_diff):2}."
                             f"{microseconds:03}  1       P\n")
            pick_offset[station] = time_diff

    return_tuple = [templates, station_dict, pick_offset]

    if time_markers:
        return_tuple.append(template_windows)

    return tuple(return_tuple)


# helper function to shift trace times
def time_Shift(trace, time_offset):
    """
    Shifts a trace in time by the specified time offset (in seconds).

    Example:
        # shift trace times by -2 seconds
        shift = -2
        shifted_trace = time_Shift(trace, shift)
    """
    frequencies = np.fft.fftfreq(trace.stats.npts, d=trace.stats.delta)
    fourier_transform = np.fft.fft(trace.data)

    # Shift
    for index, freq in enumerate(frequencies):
        fourier_transform[index] = fourier_transform[index] * np.exp(2.0
                                                                     * np.pi * 1j * freq * time_offset)

    # back fourier transform
    trace.data = np.real(np.fft.ifft(fourier_transform))
    trace.stats.starttime += time_offset

    return trace


# function to build template for matched-filter
def make_Templates(templates, template_files, station_dict, template_length,
                   template_prepick):
    """
    Generates an EQcorrscan Tribe object which stores templates for
    matched-filter analysis. Template start times are specified by the
    templates object as shown in the example, and template data are loaded
    from the miniseed files stored in files in the template_files list.
    station_dict contains the network and channel code for each station (
    information needed by EQcorrscan),

    Example:
        # create HYPODD format file with template events to import them into EQcorrscan
        # entries are: #, YR, MO, DY, HR, MN, SC, LAT, LON, DEP, MAG, EH, EZ, RMS, ID
        #              followed by lines of observations: STA, TT, WGHT, PHA
        # as specified here: https://www.ldeo.columbia.edu/~felixw/papers/Waldhauser_OFR2001.pdf
        templates = ["# 2016  9 26  9 25 49.00  61.8000 -144.0000  30.00  1.00  0.0  0.0  0.00  1\n",
                     "WASW    0.000  1       P\n",
                     "MCR3    3.000  1       P\n",
                     "N25K    3.500  1       P\n"]
        template_files = [
        "/Users/human/Dropbox/Research/Alaska/build_templates/subset_stations/AV.WASW.SHZ.2016-09-26.ms",
        "/Users/human/Dropbox/Research/Alaska/build_templates/subset_stations/YG.MCR3.BHZ.2016-09-26.ms",
        "/Users/human/Dropbox/Research/Alaska/build_templates/subset_stations/TA.N25K.BHZ.2016-09-26.ms"]

        station_dict = {"WASW": {"network": "AV", "channel": "SHZ"},
                    "MCR3": {"network": "YG", "channel": "BHZ"},
                    "N25K": {"network": "TA", "channel": "BHZ"}}

        # make templates 14 seconds with a prepick of 0.5s, so 14.5 seconds
        # total
        template_length = 14
        template_prepick = 0.5

        tribe = make_Templates(templates, files, station_dict,
                               template_length, template_prepick)
    """

    # FIXME: make this more pythonic, e.g.

    '''
    # initialize an event to add to the Obspy catalog
    event = Event(
        # define the event origin location and time
        origins=[Origin(
            latitude=61.9833, longitude=-144.0437, depth=1700, 
            time=UTCDateTime(2016, 9, 26, 8, 52, 40))],
        # define the event magnitude
        magnitudes=[Magnitude(mag=1.1)],
        # define the arrival times of phases at different stations
        picks=[
            # three picks are defined here on the BHZ component of three stations in the YG network
            Pick(time=UTCDateTime(2016, 9, 26, 8, 52, 45, 180000), phase_hint="P",
                    waveform_id=WaveformStreamID(
                        network_code="YG", station_code="RH08", channel_code="BHZ")),
            Pick(time=UTCDateTime(2016, 9, 26, 8, 52, 45, 809000), phase_hint="P",
                    waveform_id=WaveformStreamID(
                        network_code="YG", station_code="NEB1", channel_code="BHZ")),
            Pick(time=UTCDateTime(2016, 9, 26, 8, 52, 45, 661000), phase_hint="P",
                    waveform_id=WaveformStreamID(
                        network_code="YG", station_code="NEB3", channel_code="BHZ"))])
    
    # generate the catalog from a list of events (in this case 1 event comprised of 3 picks)
    catalog = Catalog([event])  
    '''

    # now write to file
    with open("templates.pha", "w") as file:
        for line in templates:
            file.write(line)

    # read the file into an Obspy catalog
    catalog = read_events("templates.pha", format="HYPODDPHA")
    # complete missing catalog info (required by EQcorrscan)
    for event in catalog.events:
        picks = event.picks.copy()
        for index, pick in enumerate(event.picks):
            if pick.waveform_id.station_code in station_dict:
                picks[index].waveform_id.network_code = \
                    station_dict[pick.waveform_id.station_code]["network"]
                picks[index].waveform_id.channel_code = \
                    station_dict[pick.waveform_id.station_code]["channel"]
                # copy Z entry
                pick_copy1 = picks[index].copy()
                pick_copy2 = picks[index].copy()
                # make N and E entries
                pick_copy1.waveform_id.channel_code = \
                    pick_copy1.waveform_id.channel_code[:-1] + 'N'
                picks.append(pick_copy1)
                pick_copy2.waveform_id.channel_code = \
                    pick_copy2.waveform_id.channel_code[:-1] + 'E'
                picks.append(pick_copy2)

        event.picks = picks

    # fig = catalog.plot(projection="local", resolution="h")

    # build stream of data from local files (use same length files as desired for detection)
    st = Stream()
    for file in template_files:
        st += read(file)

    tribe = Tribe().construct(method="from_meta_file", meta_file=catalog,
                              st=st, lowcut=1.0, highcut=15.0, samp_rate=40.0,
                              length=template_length, filt_order=4,
                              prepick=template_prepick, swin='all',
                              process_len=86400, parallel=False)  # min_snr=5.0,
    # 46 detections for 2-8 Hz
    # 56 detections for 1-15 Hz

    # print(tribe)
    return tribe


# function to drive signal detection with the help of EQcorrscan
def detect_signals(templates, template_files, station_dict, template_length,
                template_prepick, detection_files_path, start_date, end_date):
    """
    Driver function to detect signals (LFEs in this case) in time series via
    matched-filtering with specified template(s), then stacking of signals
    found from that template, and finally template matching of the stacked
    waveform template.

    Example:
        # manually define templates from station TA.N25K (location is made up)
        templates = ["# 2016  9 26  9 28 41.34  61.8000 -144.0000  30.00  1.00  0.0  0.0  0.00  1\n",
                     "N25K    0.000  1       P\n"]

        # and define a station dict to add data needed by EQcorrscan
        station_dict = {"N25K": {"network": "TA", "channel": "BHZ"}}

        # define template length and prepick length (both in seconds)
        template_length = 16.0
        template_prepick = 0.5

        # build stream of all station files for templates
        files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/N25K"
        template_files = glob.glob(f"{files_path}/*.ms")

        # define path of files for detection
        detection_files_path = "/Users/human/Desktop/alaska/inner"
        # define dates of interest
        start_date = UTCDateTime("2016-06-15T00:00:00.0Z")
        end_date = UTCDateTime("2018-08-11T23:59:59.9999999999999Z")

        # # run detection and time it
        # start = time.time()
        # party = detect_signals(templates, template_files, station_dict,
        #                        template_length, template_prepick,
        #                        detection_files_path, start_date, end_date)
        # end = time.time()
        # hours = int((end - start) / 60 / 60)
        # minutes = int(((end - start) / 60) - (hours * 60))
        # seconds = int((end - start) - (minutes * 60) - (hours * 60 * 60))
        # print(f"Runtime: {hours} h {minutes} m {seconds} s")
        # # this takes 17h 14m for 1 template of N25K across inner stations
        # # over the 2016-2018 time period

        # load party object from file
        infile = open('party_06_15_2016_to_08_12_2018.pkl', 'rb')
        party = pickle.load(infile)
        infile.close()

        # get the catalog
        catalog = party.get_catalog()

        # inspect the party growth over time
        detections_fig = party.plot(plot_grouped=True)
        rate_fig = party.plot(plot_grouped=True, rate=True)
        print(sorted(party.families, key=lambda f: len(f))[-1])

        # get the most productive family
        family = sorted(party.families, key=lambda f: len(f))[-1]
        print(family)

        # look at family template
        fig = family.template.st.plot(equal_scale=False, size=(800, 600))

        # look at family detection at index 0
        detection = family.detections[0]
        detection_time = detection.detect_time
        from figures import plot_Time_Series
        # define dates of interest
        print(f"Detection time: {detection_time}")
		doi = detection_time - 10
		doi_end = doi + 30
		# define time series files path
		files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/N25K"
		# bandpass filter from 1-15 Hz
		filter = True
		bandpass = [1, 15]
		fig = plot_Time_Series(doi, doi_end, files_path, filter=filter,
							   bandpass=bandpass)

		# TODO: plot all stations around detection times

		# TODO: stacking utilizing a small template for xcorr timeshift

	Another example spanning multiple days:
        # time the run
        import time
        start = time.time()

        # define template length and prepick length (both in seconds)
        template_length = 16.0
        template_prepick = 0.5

        # build stream of all station files for templates
        files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/2016-09-26"
        doi = UTCDateTime("2016-09-26T00:00:00.0Z")
        template_files = glob.glob(f"{files_path}/*.{doi.year}-{doi.month:02}"
                                      f"-{doi.day:02}.ms")

        # get templates and station_dict objects from picks in marker file
        marker_file_path = "lfe_templates_2.mrkr"
        prepick_offset = 11 # in seconds (was 11 for templates_2, 0 for templates_3)
        templates, station_dict, pick_offset = markers_to_template(marker_file_path, prepick_offset)

        # define path of files for detection
        detection_files_path = "/Volumes/DISK/alaska/data"
        # define period of interest for detection
        start_date = UTCDateTime("2016-06-15T00:00:00.0Z")
        end_date = UTCDateTime("2018-08-11T23:59:59.9999999999999Z")

        # # run detection
        # party = detect_signals(templates, template_files, station_dict,
        #                        template_length, template_prepick,
        #                        detection_files_path, start_date, end_date)
        # end = time.time()
        # hours = int((end - start) / 60 / 60)
        # minutes = int(((end - start) / 60) - (hours * 60))
        # seconds = int((end - start) - (minutes * 60) - (hours * 60 * 60))
        # print(f"Runtime: {hours} h {minutes} m {seconds} s")

        # load party object from file
        infile = open('party_06_15_2016_to_08_12_2018.pkl', 'rb')
        party = pickle.load(infile)
        infile.close()

        # inspect the party object detections
        detections_fig = party.plot(plot_grouped=True)
        rate_fig = party.plot(plot_grouped=True, rate=True)
        print(sorted(party.families, key=lambda f: len(f))[-1])

        # load previous stream list?
        load_stream_list = False
        # get the stacks
        start = time.time()

        stack_list = stack_waveforms(party, pick_offset, detection_files_path,
                                     template_length, template_prepick,
                                     load_stream_list=load_stream_list)

        end = time.time()
        hours = int((end - start) / 60 / 60)
        minutes = int(((end - start) / 60) - (hours * 60))
        seconds = int((end - start) - (minutes * 60) - (hours * 60 * 60))
        print(f"Runtime: {hours} h {minutes} m {seconds} s")

        # loop over stack list and show the phase weighted stack and linear
        # stack for each group
        for group in stack_list:
            # get the stack
            stack_pw = group[0]
            # get the plot handle for the stack, add a title, show figure
            plot_stack(stack_pw, filter=filter, bandpass=bandpass,
                   	   title="Phase weighted stack")

            stack_lin = group[1]
            plot_stack(stack_lin, filter=filter, bandpass=bandpass,
            		   title="Linear stack")
    """
    # set up logging (levels: DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

    tribe = make_Templates(templates, template_files, station_dict,
                           template_length, template_prepick)

    # loop over days and get detections
    iteration_date = start_date
    party_list = []
    while iteration_date < end_date:
        print(iteration_date) # print date for long runs
        # build stream of all files on day of interest
        day_file_list = glob.glob(f"{detection_files_path}/*."
                                  f"{iteration_date.year}"
                                  f"-{iteration_date.month:02}"
                                  f"-{iteration_date.day:02}.ms")

        # load files into stream
        st = Stream()
        for file in day_file_list:
            st += read(file)

        try:
            # detect
            party = tribe.detect(stream=st, threshold=8.0, daylong=True,
                                 threshold_type="MAD", trig_int=8.0, plot=False,
                                 return_stream=False, parallel_process=False,
                                 ignore_bad_data=True)
        except Exception:
            party = Party(families=[Family(template=Template())])
            pass

        # append detections to party list if there are detections
        if len(party.families[0]) > 0:
            party_list.append(party)

        # update the next file start date
        iteration_year = iteration_date.year
        iteration_month = iteration_date.month
        iteration_day = iteration_date.day
        # get number of days in current month
        month_end_day = calendar.monthrange(iteration_year,
                                            iteration_month)[1]
        # check if year should increment
        if iteration_month == 12 and iteration_day == month_end_day:
            iteration_year += 1
            iteration_month = 1
            iteration_day = 1
        # check if month should increment
        elif iteration_day == month_end_day:
            iteration_month += 1
            iteration_day = 1
        # else only increment day
        else:
            iteration_day += 1

        iteration_date = UTCDateTime(f"{iteration_year}-"
                                     f"{iteration_month:02}-"
                                     f"{iteration_day:02}T00:00:00.0Z")

    # add all detections to a single party
    if len(party_list) > 1:
        for index, iteration_party in enumerate(party_list):
            # skip first party
            if index > 0:
                # add events of current party family to first party family
                party_list[0].families[0] = party_list[0].families[0].append(
                                                   iteration_party.families[0])

    # extract and return the first party or None if no party object
    if len(party_list) > 0:
        party = party_list[0]

        # save party to pickle
        filename = f'party_{start_date.month:02}_{start_date.day:02}_' \
                   f'{start_date.year}_to_{end_date.month:02}' \
                   f'_{end_date.day:02}_{end_date.year}.pkl'
        outfile = open(filename, 'wb')
        pickle.dump(party, outfile)
        outfile.close()

        return party

    else:
        return None

    # # # # .mrkr template # # # #
    # detection over entire period
    # 601 family detections: Runtime: 47 h 44 m 34 s

    # # # 14 second template w/ 0.5 prepick # # #
    # 17 detections: "MAD" @ 11.0  <---
    # 35 detections: "MAD" @ 9.0
    # 56 detections: "MAD" @ 8.0
    # ----------------------------
    # 24 detections: "abs" @ 0.1
    # 11 detections: "abs" @ 0.12  <---
    #  3 detections: "abs" @ 0.15
    #  0 detections: "abs" @ 0.2
    # ----------------------------
    # 24 detections: "av_chan_corr" @ 0.1
    # 11 detections: "av_chan_corr" @ 0.12

    # # # 25 second template w/ 0.5 prepick # # #
    # 15 detections: "MAD" @ 11.0
    # 36 detections: "MAD" @ 9.0
    # 65 detections: "MAD" @ 8.0

    # # # 10 second template w/ 0.5 prepick # # #
    # 13 detections: "MAD" @ 11.0  <---
    # 31 detections: "MAD" @ 9.0
    # 52 detections: "MAD" @ 8.0


# function to generate linear and phase-weighted waveform stacks station by
# station (to avoid memory bottleneck) via EQcorrscan stacking routines
def stack_waveforms(party, pick_offset, streams_path, template_length,
                    template_prepick, station_dict):
    """
    Generates stacks of waveforms from families specified within a Party
    object, using the miniseed files present in the specified path
    (streams_path). Building streams for party families is slow,
    so previously generated stream lists can be used by specifying
    load_stream_list=True.

    Limitations:
        - lowest sampling rate is currently statically defined
        - limited to single day
        - filter is statically defined

    Example:
        # time the run
        import time
        start = time.time()

        # define template length and prepick length (both in seconds)
        template_length = 16.0
        template_prepick = 0.5

        # get templates and station_dict objects from picks in marker file
        marker_file_path = "lfe_templates_3.mrkr"
        prepick_offset = 0 # in seconds (was 11 for templates_2, 0 for templates_3)
        templates, station_dict, pick_offset = markers_to_template(marker_file_path, prepick_offset)

        # define path of files for detection
        detection_files_path = "/Volumes/DISK/alaska/data"

        # load party object from file
        infile = open('party_06_15_2016_to_08_12_2018.pkl', 'rb')
        party = pickle.load(infile)
        infile.close()

        # inspect the party object detections
        detections_fig = party.plot(plot_grouped=True)
        rate_fig = party.plot(plot_grouped=True, rate=True)
        print(sorted(party.families, key=lambda f: len(f))[-1])

        # # load previous stream list?
        # load_stream_list = False
        # # get the stacks station by station to avoid memory error
        # stack_list = stack_waveforms(party, pick_offset,
        #                              detection_files_path, template_length,
        #                              template_prepick, station_dict)
        # end = time.time()
        # hours = int((end - start) / 60 / 60)
        # minutes = int(((end - start) / 60) - (hours * 60))
        # seconds = int((end - start) - (minutes * 60) - (hours * 60 * 60))
        # print(f"Runtime: {hours} h {minutes} m {seconds} s")
        #
        # # save stacks as pickle file
        # outfile = open('stack_list.pkl', 'wb')
        # pickle.dump(stack_list, outfile)
        # outfile.close()

        # load stack list from file
        infile = open('stack_list.pkl', 'rb')
        stack_list = pickle.load(infile)
        infile.close()

        # loop over stack list and show the phase weighted stack and linear
        # stack for each group
        for group in stack_list:
            # get the stack
            stack_pw = group[0]
            # get the plot handle for the stack, add a title, show figure
            plot_stack(stack_pw, filter=filter, bandpass=bandpass,
                   	   title="Phase weighted stack")

            stack_lin = group[1]
            plot_stack(stack_lin, filter=filter, bandpass=bandpass,
            		   title="Linear stack")
    """
    # extract pick times for each event from party object
    # pick_times is a list of the pick times for the main trace (with
    # earliest pick time)
    pick_times = []
    for event in party.families[0].catalog.events:
        for pick in event.picks:
            # station with earliest pick defines "pick time"
            # if pick.waveform_id.station_code == "WASW" and \
            #         pick.waveform_id.network_code == "AV" and \
            #         pick.waveform_id.channel_code == "SHZ":
            #     pick_times.append(pick.time)
            # FIXME: this should be dynamic, not hard coded
            if pick.waveform_id.station_code == "WAT7" and \
                    pick.waveform_id.network_code == "AK" and \
                    pick.waveform_id.channel_code == "BHE":
                pick_times.append(pick.time)

    # loop over stations and generate a stack for each station:channel pair
    stack_pw = Stream()
    stack_lin = Stream()
    for station in station_dict.keys():
        network = station_dict[station]["network"]
        channels = []
        channels.append(station_dict[station]["channel"]) # append Z component
        channels.append(f"{channels[0][:-1]}N") # append N component
        channels.append(f"{channels[0][:-1]}E")  # append E component

        for channel in channels:
            print(f"Assembling streams for {station}.{channel}")

            stream_list = []
            for index in tqdm(range(len(pick_times))):
                pick_time = pick_times[index]

                # build stream of detection file
                day_file_list = glob.glob(f"{streams_path}/{network}."
                                          f"{station}."
                                          f"{channel}.{pick_time.year}"
                                          f"-{pick_time.month:02}"
                                          f"-{pick_time.day:02}.ms")

                # guard against missing expected files
                if len(day_file_list) > 0:
                    # FIXME: lowest samp rate should be detected, not hard coded
                    lowest_sr = 40

                    # should only be one file, but safeguard against many
                    file = day_file_list[0]

                    # extract file info from file name
                    file_station = file.split("/")[-1]
                    file_station = file_station.split(".")[1]

                    # load day file into stream
                    day_st = Stream()
                    day_st += read(file)

                    # bandpass filter
                    day_st.filter('bandpass', freqmin=1, freqmax=15)

                    # interpolate to lowest sampling rate
                    day_st.interpolate(sampling_rate=lowest_sr)

                    # match station with specified pick offset
                    station_pick_time = pick_time + pick_offset[file_station]

                    # trim trace before adding to stream from pick_offset spec
                    day_st.trim(station_pick_time, station_pick_time +
                                template_length + template_prepick)

                    stream_list.append((day_st, index))
                    # st.plot()

            # get group streams to stack (only a single group here)
            group_streams = [st_tuple[0] for st_tuple in stream_list]

            # loop over each detection in group
            for group_idx, group_stream in enumerate(group_streams):
                # align traces before stacking
                for trace_idx, trace in enumerate(group_stream):
                    # align traces from pick offset dict
                    shift = -1 * pick_offset[trace.stats.station]
                    group_streams[group_idx][trace_idx] = time_Shift(trace, shift)

            # guard against stacking error:
            try:
                # generate phase-weighted stack
                stack_pw += PWS_stack(streams=group_streams)

                # and generate linear stack
                stack_lin += linstack(streams=group_streams)

            except Exception:
                pass

    # if the stacks exist, plot them
    if len(stack_pw) > 0:
        plot_stack(stack_pw, filter=filter, bandpass=bandpass,
                   title='Phase weighted stack')

    if len(stack_lin) > 0:
        plot_stack(stack_lin, filter=filter, bandpass=bandpass,
                   title='Linear stack')

    return [stack_pw, stack_lin]

# stacking routine to generate stacks from template detections (doesn't use
# EQcorrscan stacking routine)
def stack_template_detections(party, streams_path,
                              template_length, template_prepick, station_dict,
                              main_trace, Normalize=True):
    """
    An implementation of phase-weighted and linear stacking that is
    independent of EQcorrscan routines, allowing more customization of the
    workflow.

    Example:
        # time the run
        start = time.time()

        # define template length and prepick length (both in seconds)
        template_length = 16.0
        template_prepick = 0.5

        # define the main trace to use for detections (best amplitude station)
        main_trace = ("TA", "N25K", "BHN")

        # define path of files for detection
        streams_path = "/Users/human/ak_data/inner"

        # load party object from file
        party_file = 'party_06_15_2016_to_08_12_2018.pkl'
        infile = open(party_file, 'rb')
        party = pickle.load(infile)
        infile.close()

        # inspect the party object detections
        detections_fig = party.plot(plot_grouped=True)
        rate_fig = party.plot(plot_grouped=True, rate=True)
        print(sorted(party.families, key=lambda f: len(f))[-1])

        # get the stacks
        stack_list = stack_template_detections(party, pick_offset,
                                               streams_path, template_length,
                                               template_prepick, station_dict,
                                               main_trace)
        end = time.time()
        hours = int((end - start) / 60 / 60)
        minutes = int(((end - start) / 60) - (hours * 60))
        seconds = int((end - start) - (minutes * 60) - (hours * 60 * 60))
        print(f"Runtime: {hours} h {minutes} m {seconds} s")

        # save stacks as pickle file
        outfile = open('inner_stack_list.pkl', 'wb')
        pickle.dump(stack_list, outfile)
        outfile.close()
    """
    # helper function to build a stream of detections from main_trace station
    def build_main_stream (main_trace, streams_path, pick_times):
        main_stream = Stream()
        # loop over pick times to assemble stream & show tqdm progress bar
        for index in tqdm(range(len(pick_times))):
            pick_time = pick_times[index]
            # find the local file corresponding to the station:channel pair
            file_list = glob.glob(f"{streams_path}/{main_trace[0]}."
                                      f"{main_trace[1]}."
                                      f"{main_trace[2]}.{pick_time.year}"
                                      f"-{pick_time.month:02}"
                                      f"-{pick_time.day:02}.ms")

            # guard against missing files
            if len(file_list) > 0:
                # FIXME: lowest sr should be detected, not hard coded
                lowest_sr = 40  # lowest sampling rate
                # TODO: try upsampling to 100 Hz
                # should only be one file, but safeguard against many
                file = file_list[0]
                # load day file into stream
                day_st = read(file)
                # bandpass filter
                day_st.filter('bandpass', freqmin=1, freqmax=15)
                # interpolate to lowest sampling rate
                day_st.interpolate(sampling_rate=lowest_sr)
                # trim trace to + and - 40 seconds from pick time
                day_st.trim(pick_time - 10, pick_time + 20)
                # add trace to main_stream
                main_stream += day_st

        return main_stream

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
            for trace in obspyObject:
                rms = np.sqrt(np.mean(trace.data ** 2))
                maxAmplitude = abs(trace.max())
                trace_rms.update({trace.id: rms})
                trace_maxAmplitude.update({trace.id: maxAmplitude})

            snrs = [trace_maxAmplitude[key] / trace_rms[key] for key in
                    trace_rms]

        # for Trace objects
        elif isinstance(obspyObject, obspy.core.trace.Trace):
            rms = np.sqrt(np.mean(obspyObject.data ** 2))
            maxAmplitude = abs(obspyObject.max())
            trace_rms.update({obspyObject.id: rms})
            trace_maxAmplitude.update({obspyObject.id: maxAmplitude})
            snrs = [trace_maxAmplitude[key] / trace_rms[key] for key in
                    trace_rms]

        return snrs

    # function to generate linear and phase-weighted stacks from a stream
    def generate_stacks(stream, normalize=True):
        ST = Stream()
        for tr in stream:
            if tr.data.max() > 0:
                ST += tr
        data = np.array([tr.data for tr in ST])
        if data.size == 0:
            lin = stream[0].copy()
            lin.data = np.zeros_like(lin.data)
            pws = stream[0].copy()
            pws.data = np.zeros_like(lin.data)
            return lin, pws
        data = data[
            ~np.any(np.isnan(data), axis=1)]  # remove any traces with NaN data

        if normalize:
            maxs = np.max(np.abs(data), axis=1)
            data = data / maxs[:, None]
        Linstack = data.mean(axis=0)
        phas = np.zeros_like(data)
        for ii in range(np.shape(phas)[0]):
            # hilbert transform of each timeseries
            tmp = hilbert(data[ii, :])
            # get instantaneous phase using the hilbert transform
            phas[ii, :] = np.arctan2(np.imag(tmp), np.real(tmp))
        sump = np.abs(np.sum(np.exp(np.complex(0, 1) * phas), axis=0)) / \
               np.shape(phas)[0]
        Phasestack = sump * Linstack  # traditional stack*phase stack

        lin = stream[0].copy()
        lin.data = Linstack
        pws = stream[0].copy()
        pws.data = Phasestack
        return lin, pws

    # helper function to determine time offset of each time series in a
    # stream with respect to a main trace via cross correlation
    def xcorr_time_shifts(stream, shift_len=10):
        shifts = []
        indices = []

        # find reference index with strongest signal, this serves as a template
        max_snr = 0
        for index, trace in enumerate(stream):
            trace_snr = snr(trace)[0]
            if trace_snr > max_snr:
                max_snr = trace_snr
                reference_idx = index
                # get the time associated with the maximum amplitude signal
                max_amplitude_value, max_amplitude_index = max_amplitude(trace)
                max_amplitude_offset = max_amplitude_index / \
                                       trace.stats.sampling_rate
                # trim the reference trace to + and - 1.5 seconds
                # surrounding max amplitude signal
                reference_start_time = trace.stats.starttime + \
                                       max_amplitude_offset - 1.5
                reference_trace = trace.copy().trim(reference_start_time,
                                                    reference_start_time + 3)

        if max_snr == 0:
            print("Error: max snr is 0")
        else:
            print(f"Max snr in main_trace template: {max_snr}")

        # loop through each trace and get cross-correlation time delay
        for st_idx, trace in enumerate(stream):
            # case: all traces that are not the template trace
            if st_idx != reference_idx:
                # correlate the reference trace through the trace
                cc = correlate_template(trace, reference_trace, mode='valid',
                                        normalize='naive', demean=True,
                                        method='auto')
                # find the index with the max correlation coefficient
                max_idx = np.argmax(cc)

                # # to visualize a trace, the template, and the max correlation
                # stt = Stream()
                # stt += trace # the trace
                # # the section of the trace where max correlation coef. starts
                # stt += trace.copy().trim(trace.stats.starttime + (max_idx /
                #                          trace.stats.sampling_rate),
                #                          trace.stats.endtime)
                # # the template aligned with the max correlation section
                # stt += reference_trace.copy()
                # stt[2].stats.starttime = stt[1].stats.starttime
                # stt.plot()

                # keep track of negative correlation coefficients
                if cc.max() < 0:
                    indices.append(st_idx)

                # TODO: - - - WORKING HERE - - -
                # append the cross correlation time shift for this trace
                shifts.append(max_idx / trace.stats.sampling_rate)

                # to visualize a trace, the template, and the max correlation
                stt = Stream()
                stt += trace # the trace
                # the section of the trace where max correlation coef. starts
                stt += trace.copy().trim(trace.stats.starttime + (max_idx /
                                         trace.stats.sampling_rate),
                                         trace.stats.endtime)
                # the template aligned with the max correlation section
                stt += reference_trace.copy()
                stt[2].stats.starttime = stt[1].stats.starttime
                stt.plot()


            # case: zero time shift for the template trace
            else:
                shifts.append(0)

        return shifts, indices

    # helper function to align all traces in a stream based on xcorr shifts
    def align_stream(stream, shifts):
        group_streams = Stream()
        # define a maximum time shift (3x the largest xcorr shift)
        max_shift = max(3 * abs(np.asarray(shifts)))
        # shift each trace and append to new Stream
        for tr_idx, tr in enumerate(stream):
            ref_time = tr.stats.starttime
            # TODO: trace what is going on below with time shifts
            tr_copy = tr.copy().trim(tr.stats.starttime - (max_shift +
                                     shifts[tr_idx]), tr.stats.endtime +
                                     max_shift - shifts[tr_idx], pad=True,
                                     fill_value=0)
            tr_copy.stats.starttime = stream[0].stats.starttime - max_shift
            tr_copy.trim(tr_copy.stats.starttime + 1, tr_copy.stats.endtime -
                         1, pad=True, fill_value=0)
            tr_copy.stats.starttime = ref_time
            group_streams += tr_copy
        return group_streams

    # first extract pick times for each event from party object
    # pick_times is a list of the pick times for the main trace
    pick_times = []
    pick_network, pick_station, pick_channel = main_trace
    for event in party.families[0].catalog.events:
        for pick in event.picks:
            # trace with highest amplitude signal
            if pick.waveform_id.station_code == pick_station and \
                    pick.waveform_id.network_code == pick_network and \
                    pick.waveform_id.channel_code == pick_channel:
                pick_times.append(pick.time)

    # build dict of stations from contents of specified directory
    file_list = glob.glob(f"{streams_path}/*.ms")
    station_dict = {}
    for file in file_list:
        filename = file.split("/")[-1]
        file_station = filename.split(".")[1]
        if file_station not in station_dict:
            file_network = filename.split(".")[0]
            file_channel = filename.split(".")[2][:-1] + "Z" # force z component
            station_dict[file_station] = {"network": file_network,
                                          "channel": file_channel}

    # get time shifts associated with detections on main trace
    # TODO: WORKING HERE
    pick_times = pick_times[:100]
    main_stream = build_main_stream(main_trace, streams_path, pick_times)
    plot_stack(main_stream)
    shifts, _ = xcorr_time_shifts(main_stream)

    # loop over stations and generate a stack for each station:channel pair
    stack_pw = Stream()
    stack_lin = Stream()
    for station in station_dict.keys():
        network = station_dict[station]["network"]
        channels = []
        channels.append(station_dict[station]["channel"])  # append Z component
        channels.append(f"{channels[0][:-1]}N")  # append N component
        channels.append(f"{channels[0][:-1]}E")  # append E component

        for channel in channels:
            print(f"Assembling streams for {station}.{channel}")

            sta_chan_stream = Stream()
            for index in tqdm(range(len(pick_times))):
                pick_time = pick_times[index]

                # find the local file corresponding to the station:channel pair
                day_file_list = glob.glob(f"{streams_path}/{network}."
                                          f"{station}."
                                          f"{channel}.{pick_time.year}"
                                          f"-{pick_time.month:02}"
                                          f"-{pick_time.day:02}.ms")

                # guard against missing files
                if len(day_file_list) > 0:
                    # FIXME: lowest sr should be detected, not hard coded
                    lowest_sr = 40 # lowest sampling rate
                    # TODO: try upsampling to 100 Hz

                    # should only be one file, but safeguard against many
                    file = day_file_list[0]

                    # load day file into stream
                    day_st = read(file)

                    # bandpass filter
                    day_st.filter('bandpass', freqmin=1, freqmax=15)

                    # interpolate to lowest sampling rate
                    day_st.interpolate(sampling_rate=lowest_sr)

                    # trim trace to + and - 40 seconds from pick time
                    day_st.trim(pick_time - 20, pick_time + 50)

                    sta_chan_stream += day_st

            # guard against empty stream
            if len(sta_chan_stream) > 0:
                # get xcorr time shift
                shifts, indices = xcorr_time_shifts(sta_chan_stream)

                # align each trace in stream based on specified time shifts
                aligned_sta_chan_stream = align_stream(sta_chan_stream, shifts)

                # guard against stacking error:
                try:
                    # generate linear and phase-weighted stack
                    lin, pws = generate_stacks(aligned_sta_chan_stream)
                    # add phase-weighted stack to stream
                    stack_pw += pws
                    # and add linear stack to stream
                    stack_lin += lin

                except Exception:
                    pass

    # if the stacks exist, plot them and don't bandpass filter from 1-15 Hz
    filter = False
    bandpass = [1, 15]
    if len(stack_pw) > 0:
        plot_stack(stack_pw, filter=False, bandpass=bandpass,
                   title='Phase weighted stack', save=True)

    if len(stack_lin) > 0:
        plot_stack(stack_lin, filter=False, bandpass=bandpass,
                   title='Linear stack', save=True)

    return [stack_pw, stack_lin]


# function for matched-filtering of stacked templates through time series
# autocorrelation
def template_match_stack():
    """

    Example:
        # load stack list from file
        infile = open('alt_stack_list.pkl', 'rb')
        stack_list = pickle.load(infile)
        infile.close()

    """
    # shift_len?
    # should template be filtered already? deal with this for detect

    # some basic check for false detections? (constant slope line in cumulative)

    pass

