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
                              process_len=86400, parallel=True)  # min_snr=5.0,
    # 46 detections for 2-8 Hz
    # 56 detections for 1-15 Hz

    # print(tribe)
    return tribe


# function to drive LFE detection with EQcorrscan
def detect_LFEs(templates, template_files, station_dict, template_length,
                template_prepick, detection_files_path, doi):
    """
    Driver function to detect LFEs in time series via matched-filtering with a
    specified template, stacking of LFEs found from that template,
    and template matching of the stacked waveform template.

    Example:
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
        template_length = 25.0
        template_prepick = 0.5

        # build stream of all station files for templates
        files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/picked"
        # files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/data"
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
        doi = UTCDateTime("2016-09-26T00:00:00.0Z")

        # run detection
        party = detect_LFEs(templates, template_files, station_dict,
                            template_length, template_prepick,
                            detection_files_path, doi)

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
    """
    # set up logging
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

    tribe = make_Templates(templates, template_files, station_dict,
                           template_length, template_prepick)

    # FIXME: detect on multiple days

    # build stream of all stations for detection
    day_file_list = glob.glob(f"{detection_files_path}/*.{doi.year}"
                              f"-{doi.month:02}"
                              f"-{doi.day:02}.ms")
    # load files into stream
    st = Stream()
    for file in day_file_list:
        st += read(file)

    # detect
    party = tribe.detect(stream=st, threshold=9.0, daylong=True,
                         threshold_type="MAD", trig_int=12.0, plot=True,
                         return_stream=False, parallel_process=True)

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
    # 15 detections: "MAD" @ 11.0  <---
    #  detections: "MAD" @ 9.0
    #  detections: "MAD" @ 8.0

    return party


# function to generate phase-weighted waveform stack
def stack_Waveforms(party, pick_offset, streams_path, template_length,
                    template_prepick, load_stream_list=False):
    f"""
    Generates stacks of waveforms from families specified within a Party
    object (as returned by detect_LFEs), using the miniseed files present in
    the specified path (streams_path). Building streams for party families
    is slow, so previously generated stream lists can be used by specifying
    load_stream_list=True.

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
        stack_list = stack_Waveforms(party, pick_offset, streams_path,
                                     template_length, template_prepick,
                                     load_stream_list=load_stream_list)

        # loop over stack list and print the phase weighted stack and linear
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
            if pick.waveform_id.station_code == "WASW" and \
                    pick.waveform_id.network_code == "AV" and \
                    pick.waveform_id.channel_code == "SHZ":
                pick_times.append(pick.time)

    if not load_stream_list:
        # build streams from party families
        stream_list = []
        for index, pick_time in enumerate(pick_times):

            # build stream of all stations for detection
            day_file_list = glob.glob(f"{streams_path}/*.{pick_time.year}"
                                      f"-{pick_time.month:02}"
                                      f"-{pick_time.day:02}.ms")
            # load files into stream
            st = Stream()
            lowest_sr = 40
            for file in day_file_list:
                # extract file info from file name
                file_station = file[60:].split(".")[1]

                # load day file into stream
                day_tr = read(file)

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

    # cluster via xcorr
    # groups = cluster(template_list=stream_list, show=True, corr_thresh=0.1, cores=2)
    # groups[0][0][0].plot()

    # or build a single group manually
    groups = [stream_list]

    stack_list = []
    # loop over each group of detections (from clustering)
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

# function for matched-filtering of stacked templates through time series
def matched_Filter_Stack():
    """

    Returns:

    """

    pass