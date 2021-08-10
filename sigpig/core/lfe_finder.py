"""
Functions to find low-frequency earthquakes in seismic time series data.
"""

import logging
from obspy import UTCDateTime, Stream, Trace, read, read_events
from eqcorrscan import Tribe
from eqcorrscan.utils.clustering import cluster
from eqcorrscan.utils.stacking import PWS_stack, linstack, align_traces
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import calendar
from tqdm import tqdm


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

    # create pick offset dict for pick offset from master trace
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


# function to drive LFE detection with EQcorrscan
def detect_LFEs(templates, template_files, station_dict, template_length,
                template_prepick, detection_files_path, start_date, end_date):
    """
    Driver function to detect LFEs in time series via matched-filtering with a
    specified template, stacking of LFEs found from that template,
    and template matching of the stacked waveform template.

    Example:
        # manually define templates
        templates = ["# 2016  9 26  9 25 46.00  61.8000 -144.0000  30.00  1.00  0.0  0.0  0.00  1\n",
                     "GLB     4.000  1       P\n",
                     "PTPK   19.000  1       P\n",
                     "WASW    0.000  1       P\n",
                     "MCR4    7.000  1       P\n",
                     "NEB3    8.500  1       P\n",
                     "MCR1    8.500  1       P\n",
                     "RH08   16.500  1       P\n",
                     "RH10   15.500  1       P\n",
                     "RH09   15.500  1       P\n",
                     "WACK    3.500  1       P\n",
                     "NEB1   10.500  1       P\n",
                     "N25K    3.500  1       P\n",
                     "MCR3    3.500  1       P\n",
                     "KLU    21.000  1       P\n",
                     "MCR2    1.500  1       P\n"]

        # define template length and prepick length (both in seconds)
        template_length = 16.0
        template_prepick = 0.5

        # build stream of all station files for templates
        files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/picked"
        doi = UTCDateTime("2016-09-26T00:00:00.0Z")
        template_files = glob.glob(f"{files_path}/*.{doi.year}-{doi.month:02}"
                                      f"-{doi.day:02}.ms")

        # station dict to add data needed by EQcorrscan
        station_dict = {"GLB": {"network": "AK", "channel": "BHZ"},
                        "PTPK": {"network": "AK", "channel": "BHZ"},
                        "WASW": {"network": "AV", "channel": "SHZ"},
                        "MCR4": {"network": "YG", "channel": "BHZ"},
                        "NEB3": {"network": "YG", "channel": "BHZ"},
                        "MCR1": {"network": "YG", "channel": "BHZ"},
                        "RH08": {"network": "YG", "channel": "BHZ"},
                        "RH10": {"network": "YG", "channel": "BHZ"},
                        "RH09": {"network": "YG", "channel": "BHZ"},
                        "WACK": {"network": "AV", "channel": "BHZ"},
                        "NEB1": {"network": "YG", "channel": "BHZ"},
                        "N25K": {"network": "TA", "channel": "BHZ"},
                        "MCR3": {"network": "YG", "channel": "BHZ"},
                        "KLU": {"network": "AK", "channel": "BHZ"},
                        "MCR2": {"network": "YG", "channel": "BHZ"}}

        # define path of files for detection
        detection_files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/picked"
        # define day of interest
        start_date = UTCDateTime("2016-09-26T00:00:00.0Z")
        end_date = UTCDateTime("2016-09-26T23:59:59.999999999999Z")

        # run detection
        party = detect_LFEs(templates, template_files, station_dict,
                            template_length, template_prepick,
                            detection_files_path, start_date, end_date)

        # # to inspect the catalog
        # catalog = party.get_catalog()

        # inspect the party object
        fig = party.plot(plot_grouped=True)

        # peek at most productive family
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
		files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/picked"
		# bandpass filter from 1-15 Hz
		filter = True
		bandpass = [1, 15]
		fig = plot_Time_Series(doi, doi_end, files_path, filter=filter,
							   bandpass=bandpass)

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
        marker_file_path = "lfe_template.mrkr"
        prepick_offset = 11 # in seconds
        templates, station_dict, pick_offset = markers_to_template(marker_file_path, prepick_offset)

        # define path of files for detection
        detection_files_path = "/Volumes/DISK/alaska/data"
        # define period of interest for detection
        start_date = UTCDateTime("2016-06-15T00:00:00.0Z")
        end_date = UTCDateTime("2018-08-11T23:59:59.9999999999999Z")

        # # run detection
        # party = detect_LFEs(templates, template_files, station_dict,
        #                     template_length, template_prepick,
        #                     detection_files_path, start_date, end_date)
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

        # stack_list = stack_waveforms(party, pick_offset, detection_files_path,
        #                              template_length, template_prepick,
        #                              load_stream_list=load_stream_list)

        stack_list = stack_waveforms_1x1(party, pick_offset,
                                         detection_files_path, template_length,
                                         template_prepick, station_dict)

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
            pw_fig = stack_pw.plot(handle=True)
            pw_fig.suptitle(f"Phase weighted stack: Template={template_length}, Prepick={template_prepick}")

            stack_lin = group[1]
            lin_fig = stack_lin.plot(handle=True)
            lin_fig.suptitle(f"Linear stack: Template={template_length}, Prepick={template_prepick}")
            plt.show()
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

        # detect
        party = tribe.detect(stream=st, threshold=8.0, daylong=True,
                             threshold_type="MAD", trig_int=8.0, plot=False,
                             return_stream=False, parallel_process=False,
                             ignore_bad_data=True)

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


# function to generate phase-weighted waveform stack all at once (memory
# limited)
def stack_waveforms(party, pick_offset, streams_path, template_length,
                    template_prepick, load_stream_list=False):
    """
    Generates stacks of waveforms from families specified within a Party
    object (as returned by detect_LFEs), using the miniseed files present in
    the specified path (streams_path). Building streams for party families
    is slow, so previously generated stream lists can be used by specifying
    load_stream_list=True.

    Limitations:
        - lowest sampling rate is currently statically defined
        - limited to single day
        - filter is statically defined

    Example:
        # get detections (see example for detect_LFEs) 
        party = detect_LFEs(templates, template_files, station_dict,
                            template_length, template_prepick,
                            detection_files_path, doi)

        # create pick offset dict for pick offset from master trace
        pick_offset = {"GLB": 4.0, "PTPK": 19.0, "WASW": 0.0, "MCR4": 7.0,
                       "NEB3": 8.5, "MCR1": 8.5, "RH08": 16.5, "RH10": 15.5,
                       "RH09": 15.5, "WACK": 3.5, "NEB1": 10.5, "N25K": 3.5,
                       "MCR3": 3.5, "KLU": 21.0, "MCR2": 1.5}

        # define path where miniseed files are stored
        streams_path = "/Users/human/Dropbox/Research/Alaska/build_templates/picked"

        # load previous stream list?
        load_stream_list = False
        # get the stacks
        stack_list = stack_waveforms(party, pick_offset, streams_path,
                                     template_length, template_prepick,
                                     load_stream_list=load_stream_list)

        # loop over stack list and show the phase weighted stack and linear
        # stack for each group
        for group in stack_list:
            # get the stack
            stack_pw = group[0]
            # get the plot handle for the stack, add a title, show figure
            pw_fig = stack_pw.plot(handle=True)
            pw_fig.suptitle(f"Phase weighted stack: Template={template_length}, Prepick={template_prepick}")

            stack_lin = group[1]
            lin_fig = stack_lin.plot(handle=True)
            lin_fig.suptitle(f"Linear stack: Template={template_length}, Prepick={template_prepick}")
            plt.show()
    """
    # extract pick times for each event from party object
    # pick_times is a list of the pick times for the master trace (with
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

    if not load_stream_list:
        # build streams from party families
        stream_list = []
        for index, pick_time in enumerate(pick_times):
            print(index)

            # build stream of all stations for detection
            day_file_list = glob.glob(f"{streams_path}/*.{pick_time.year}"
                                      f"-{pick_time.month:02}"
                                      f"-{pick_time.day:02}.ms")
            # load files into stream
            st = Stream()
            # FIXME: this should be detected, not hard coded
            lowest_sr = 40
            for file in day_file_list:
                # extract file info from file name
                # FIXME: this should be dynamic, not hard coded
                file_station = file[26:].split(".")[1]

                if file_station in pick_offset.keys():
                    # load day file into stream
                    day_tr = Stream()
                    day_tr += read(file)

                    # bandpass filter
                    day_tr.filter('bandpass', freqmin=1, freqmax=15)

                    # interpolate to lowest sampling rate
                    day_tr.interpolate(sampling_rate=lowest_sr)

                    # match station with specified pick offset
                    station_pick_time = pick_time + pick_offset[file_station]

                    # trim trace before adding to stream from pick_offset spec
                    day_tr.trim(station_pick_time, station_pick_time +
                                template_length + template_prepick)

                    st += day_tr

                    # # check if lowest sampling rate
                    # if st[0].stats.sampling_rate < lowest_sr:
                    #     lowest_sr = st[0].stats.sampling_rate

            stream_list.append((st, index))
            # st.plot()

        # save stream list as pickle file
        outfile = open('stream_list.pkl', 'wb')
        pickle.dump(stream_list, outfile)
        outfile.close()

    else:
        # load stream list from file
        infile = open('stream_list.pkl', 'rb')
        stream_list = pickle.load(infile)
        infile.close()

    print("finished making stream")

    # cluster via xcorr
    # groups = cluster(template_list=stream_list, show=True, corr_thresh=0.1, cores=2)
    # groups[0][0][0].plot()

    # or build a single group manually
    groups = [stream_list]

    stack_list = []
    # loop over each group of detections (from clustering or manual assignment)
    for group in groups:
        # get group streams to stack
        group_streams = [st_tuple[0] for st_tuple in group]

        # loop over each detection in group
        for group_idx, group_stream in enumerate(group_streams):
            # align traces before stacking
            for trace_idx, trace in enumerate(group_stream):
                # align traces from pick offset dict
                shift = -1 * pick_offset[trace.stats.station]
                group_streams[group_idx][trace_idx] = time_Shift(trace, shift)

        # generate phase-weighted stack
        stack_pw = PWS_stack(streams=group_streams)

        # or generate linear stack
        stack_lin = linstack(streams=group_streams)

        stack_list.append([stack_pw, stack_lin])

    return stack_list

# function to generate phase-weighted waveform stack station by station
def stack_waveforms_1x1(party, pick_offset, streams_path, template_length,
                    template_prepick, station_dict):
    """
    Generates stacks of waveforms from families specified within a Party
    object (as returned by detect_LFEs), using the miniseed files present in
    the specified path (streams_path). Building streams for party families
    is slow, so previously generated stream lists can be used by specifying
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
        marker_file_path = "lfe_template.mrkr"
        prepick_offset = 11 # in seconds
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

        # load previous stream list?
        load_stream_list = False
        # get the stacks station by station to avoid memory error
        stack_list = stack_waveforms_1x1(party, pick_offset,
                                         detection_files_path, template_length,
                                         template_prepick, station_dict)
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
            pw_fig = stack_pw.plot(handle=True)
            pw_fig.suptitle(f"Phase weighted stack: Template={template_length}, Prepick={template_prepick}")

            stack_lin = group[1]
            lin_fig = stack_lin.plot(handle=True)
            lin_fig.suptitle(f"Linear stack: Template={template_length}, Prepick={template_prepick}")
            plt.show()
    """
    # extract pick times for each event from party object
    # pick_times is a list of the pick times for the master trace (with
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
                    # FIXME: this should be detected, not hard coded
                    lowest_sr = 40

                    # should only be one file, but safeguard against many
                    file = day_file_list[0]

                    # extract file info from file name
                    # FIXME: this should be dynamic, not hard coded
                    file_station = file[26:].split(".")[1]

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

                return [stack_pw, stack_lin]

            except Exception:
                pass

    return [stack_pw, stack_lin]


# function for matched-filtering of stacked templates through time series
def matched_filter_stack():
    """

    Example:


    """

    pass

