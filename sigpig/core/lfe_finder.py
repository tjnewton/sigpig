"""
Functions to find low-frequency earthquakes in time series. 
"""

import logging
from obspy import UTCDateTime, Stream, read, read_events
from obspy.core.event import Origin
from eqcorrscan import Tribe
from eqcorrscan.utils.clustering import cluster
from eqcorrscan.utils.stacking import PWS_stack, linstack
import glob
import os

# function to build template for matched-filter
def make_Templates(templates, template_files, station_dict):
    """
    # TODO:

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

        tribe = make_Templates(templates, files, station_dict)
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

    tribe = Tribe().construct(
        method="from_meta_file", meta_file=catalog, st=st, lowcut=1.0,
        highcut=15.0,
        samp_rate=40.0, length=14.0, filt_order=4, prepick=0.5, swin='all',
        process_len=86400, parallel=True)  # min_snr=5.0,
    # 46 detections for x2-8 Hz
    # 56 detections for 1-15 Hz

    # print(tribe)
    return tribe

# function to drive LFE detection with EQcorrscan
def detect_LFEs(templates, template_files, station_dict,
                detection_files_path, doi):
    """
    # TODO:

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
                                detection_files_path, doi)

        # to inspect the catalog
        catalog = party.get_catalog()

        # inspect the party object
        fig = party.plot(plot_grouped=True)

        # peek at most productive family
        family = sorted(party.families, key=lambda f: len(f))[-1]
        print(family)
        fig = family.template.st.plot(equal_scale=False, size=(800, 600))

        # look at first family detection
        detection_1 = family.detections[27]
        detection_1_time = detection_1.detect_time
        from figures import plot_Time_Series
        # define dates of interest
		doi = detection_1_time - 40
		doi_end = doi + 80
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

    tribe = make_Templates(templates, template_files, station_dict)

    # build stream of all stations for detection
    day_file_list = glob.glob(f"{detection_files_path}/*.{doi.year}"
                              f"-{doi.month:02}"
                              f"-{doi.day:02}.ms")
    # load files into stream
    st = Stream()
    for file in day_file_list:
        st += read(file)

    # detect
    party = tribe.detect(stream=st, threshold=8.0, daylong=True,
                             threshold_type="MAD", trig_int=12.0,
                             plot=True, return_stream=False,
                             parallel_process=True)

    return party

# function to generate phase-weighted waveform stack
def stack_Waveforms(party, streams_path):
    """
    # TODO:

    Example:
        streams_path = "/Users/human/Dropbox/Research/Alaska/build_templates/picked"


    """
    # # get the catalog
    # catalog = party.get_catalog()
    #
    # # store the information necessary for EQcorrscan clustering
    # origin = Origin()
    # origin.latitude = 61.8000
    # origin.longitude = -144.0000
    # origin.depth = 30.0
    # # set origin for all events
    # for index, _ in enumerate(catalog.events):
    #     catalog.events[index].origins.append(origin)

    # extract pick times for each event from party object
    pick_times = []
    for event in party.families[0].catalog.events:
        for pick in event.picks:
            # station with earliest pick defines "pick time"
            if pick.waveform_id.station_code == "WASW" and \
                    pick.waveform_id.network_code == "AV" and \
                    pick.waveform_id.channel_code == "SHZ":
                pick_times.append(pick.time)

    # build streams from party families
    stream_list = []
    for index, pick_time in enumerate(pick_times):

        # build stream of all stations for detection
        day_file_list = glob.glob(f"{streams_path}/*.{pick_time.year}"
                                  f"-{pick_time.month:02}"
                                  f"-{pick_time.day:02}.ms")
        # load files into stream
        st = Stream()
        lowest_sr = 10000
        for file in day_file_list:
            st += read(file)

            # check if lowest sampling rate
            if st[0].stats.sampling_rate < lowest_sr:
                lowest_sr = st[0].stats.sampling_rate

        # interpolate to lowest sampling rate
        st.interpolate(sampling_rate=lowest_sr)

        # trim streams to time period of interest
        st.trim(pick_time - 10, pick_time + 50)

        stream_list.append((st, index))

    # # gather stream file list from specified directory
    # stream_files = glob.glob(os.path.join(templates_path, '*'))
    # stream_list = [(read(stream_file), i) for i, stream_file in enumerate(
    #                stream_files)]

    # for st in stream_list:
    #     st[0].plot()

    # # cluster via xcorr values
    # groups = cluster(template_list=stream_list, show=True, corr_thresh=0.1,
    #                  cores=2)

    # build groups manually
    groups = []

    # get group streams to stack
    group_streams = [st_tuple[0] for st_tuple in groups[0]]

    # align traces before stacking

    # generate phase-weighted stack
    stack = PWS_stack(streams=group_streams)
    lin_stack = linstack(streams=group_streams)


    return # FIXME