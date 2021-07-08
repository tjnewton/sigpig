"""
Functions to find low-frequency earthquakes in time series. 
"""

import logging
from obspy import UTCDateTime, Stream, read, read_events
from eqcorrscan import Tribe
import glob

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
        for pick in event.picks:
            if pick.waveform_id.station_code in station_dict:
                pick.waveform_id.network_code = \
                station_dict[pick.waveform_id.station_code]["network"]
                pick.waveform_id.channel_code = \
                station_dict[pick.waveform_id.station_code]["channel"]

    # fig = catalog.plot(projection="local", resolution="h")

    # build stream of data from local files (use same length files as desired for detection)
    st = Stream()
    for file in template_files:
        st += read(file)

    tribe = Tribe().construct(
        method="from_meta_file", meta_file=catalog, st=st, lowcut=1.0,
        highcut=10.0,
        samp_rate=40.0, length=14.0, filt_order=4, prepick=0.5, swin='all',
        process_len=86400, parallel=True)  # min_snr=5.0,

    # print(tribe)
    return tribe

# function to drive LFE detection with EQcorrscan
def detect_LFEs(templates, template_files, station_dict,
                detection_files_path, doi):
    """
    # TODO:

    Example:
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

        # define path of files for detection
        detection_files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/subset_stations"
        # define day of interest
        doi = UTCDateTime("2016-09-26T00:00:00.0Z")

        # run detection
        party, st = detect_LFEs(templates, template_files, station_dict,
                                detection_files_path, doi)

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
    party, st = tribe.detect(stream=st, threshold=9.0, daylong=True,
                             threshold_type="MAD", trig_int=4.0,
                             plot=False, return_stream=True,
                             parallel_process=True)

    return party, st
