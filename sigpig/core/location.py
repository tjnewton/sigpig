"""
Functions to constrain the origin location of signals.
"""
# stop numpy using all available threads (these environment variables must be
# set before numpy is imported for the first time).
import os
import utm
os.environ.update(OMP_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1",
                  NUMEXPR_NUM_THREADS="1",
                  MKL_NUM_THREADS="1")
import pathlib
from obspy import read, UTCDateTime, Inventory
from obspy.clients.fdsn import Client
import glob
from obspy.core import AttribDict
from pyproj import Proj
from quakemigrate import QuakeScan, Trigger
from quakemigrate.io import Archive, read_lut, read_stations, read_vmodel, \
                            read_response_inv
from quakemigrate.signal.onsets import STALTAOnset
from quakemigrate.lut import compute_traveltimes
from quakemigrate.signal.pickers import GaussianPicker
from quakemigrate.signal.local_mag import LocalMag

import pandas as pd
from scipy.io import savemat
import numpy as np
import datetime
import matplotlib.pyplot as plt
import math
from data import rattlesnake_Ridge_Station_Locations

def download_date():
    """
    This script will download the waveform data and an instrument response
    inventory from IRIS (in miniSEED and STATIONXML formats, respectively)
    for the Iceland dike intrusion example.

    """

    # --- i/o paths ---
    station_file = "./inputs/wrangell_stations.txt"
    data_path = pathlib.Path("./inputs/mSEED")
    response_file = "./inputs/resp.xml"

    # --- Set network code & client ---
    networks = ['AK', 'AT', 'AV', 'NP', 'TA', 'YG']
    network_stations = {"AK": ["BAL", "BARN", "DHY", "DIV", "DOT", "GHO", "GLB",
                               "K218", "KLU", "KNK", "MCAR", "MCK", "PAX", "PTPK",
                               "RIDG", "RND", "SAW", "SCM", "VRDI", "WAT1",
                               "WAT6", "WAT7"],
                        "AT": ["MENT", "PMR"],
                        "AV": ["WACK", "WASW", "WAZA"],
                        "NP": ["2730", "2738", "2784", "8034", "AJKS", "AMJG"],
                        "TA": ["HARP", "K24K", "L26K", "L27K", "M23K", "M24K",
                               "M26K", "M27K", "N25K"],
                        "YG": ["DEN1", "DEN3", "DEN4", "DEN5", "GLN1", "GLN2",
                               "GLN3", "GLN4", "LKLO", "MCR1", "MCR2", "MCR3",
                               "MCR4", "NEB1", "NEB2", "NEB3", "RH01", "RH02",
                               "RH03", "RH04", "RH05", "RH06", "RH07", "RH08",
                               "RH09", "RH10", "RH11", "RH12", "RH13", "RH14",
                               "RH15", "TOK1", "TOK2", "TOK3", "TOK4", "TOK5"]}
    network_channels = {"AK": ["BH*", "BN*", "HN*"], "AT": ["BH*"], "AV": [
                        "BH*", "SH*"], "NP": ["HN*"], "TA": ["BH*"], "YG": ["BH*"]}

    # --- Set time period ---
    starttime = UTCDateTime("2016-09-26T15:40:00")
    endtime = UTCDateTime("2016-09-26T15:50:00")

    #  --- Read in station file ---
    stations = read_stations(station_file)

    # --- Download instrument response inventory ---
    datacentre = "IRIS"
    client = Client(datacentre)
    inv = Inventory()
    for network in networks:
        for station in network_stations[network]:
            try:
                inv += client.get_stations(network=network, station=station,
                                           starttime=starttime, endtime=endtime,
                                           level="response")
            except Exception:
                pass
    inv.write(response_file, format="STATIONXML")

    # --- Make directories to store waveform data ---
    waveform_path = data_path / str(starttime.year) / f"{starttime.julday:03d}"
    waveform_path.mkdir(parents=True, exist_ok=True)

    # --- Save waveform data subset ---
    for network in networks:
        for station in network_stations[network]:
            # # load stream from daily waveform file
            # file_list = glob.glob(f"../build_templates/data/{network}."
            #                       f"{station}.***.{starttime.year}-"
            #                       f"{starttime.month:02}-{starttime.day:02}.ms")
            # # only consider first entry if multiple channels
            # st = read(file_list[0])

            for channel in network_channels[network]:
                try:
                    print(f"Downloading waveform data for station {station} from {datacentre}")
                    st = client.get_waveforms(network=network, station=station,
                                              location="*", channel=channel,
                                              starttime=starttime, endtime=endtime)

                    st.merge(method=-1)
                    for comp in ["E", "N", "Z"]:
                        try:
                            st_comp = st.select(component=comp)
                            st_comp.write(str(waveform_path / f"{station}_{comp}.m"),
                                          format="MSEED")
                        except IndexError:
                            pass

                except Exception:
                    pass


def generate_look_up_table():
    """
    This script generates the traveltime look-up table (LUT) for the Iceland dike
    intrusion example.

    """
    # --- i/o paths ---
    station_file = "./inputs/wrangell_stations.txt"
    vmodel_file = "./inputs/wrangell_vmodel.txt"
    lut_out = "./outputs/lut/wrangell.LUT"

    # --- Read in the station information file ---
    stations = read_stations(station_file)

    # --- Read in the velocity model file ---
    vmodel = read_vmodel(vmodel_file)

    # --- Define the input and grid projections ---
    gproj = Proj(proj="lcc", units="km", lon_0=-145.18, lat_0=62.14,
                 lat_1=61.21,
                 lat_2=63.07, datum="WGS84", ellps="WGS84", no_defs=True)
    cproj = Proj(proj="longlat", datum="WGS84", ellps="WGS84", no_defs=True)

    # --- Define the grid specifications ---
    # AttribDict behaves like a Python dict, but also has '.'-style access.
    grid_spec = AttribDict()
    grid_spec.ll_corner = [-149.3515, 60.2834, -2.6]
    grid_spec.ur_corner = [-141.0205, 663.9980, 100.0]
    grid_spec.node_spacing = [1.0, 1.0, 0.5]
    grid_spec.grid_proj = gproj
    grid_spec.coord_proj = cproj

    # --- 1-D velocity model LUT generation (using NonLinLoc eikonal solver) ---
    lut = compute_traveltimes(grid_spec, stations, method="1dnlloc",
                              vmod=vmodel,
                              phases=["P", "S"], log=True, save_file=lut_out)


def detect_signals():
    """
    This script runs the detect stage for the Iceland dike intrusion example.

    """

    # --- i/o paths ---
    station_file = "./inputs/wrangell_stations.txt"
    data_in = "./inputs/mSEED"
    lut_file = "./outputs/lut/wrangell.LUT"
    run_path = "./outputs/runs"

    # --- Set time period over which to run detect ---
    starttime = "2016-09-26T15:40:00"
    endtime = "2016-09-26T15:50:00"
    run_name = f"{starttime[:13]}-{starttime[14:16]}_run"

    # --- Read in station file ---
    stations = read_stations(station_file)

    # --- Create new Archive and set path structure ---
    archive = Archive(archive_path=data_in, stations=stations,
                      archive_format="YEAR/JD/STATION")

    # --- Load the LUT ---
    lut = read_lut(lut_file=lut_file)
    lut.decimate([2, 2, 2], inplace=True)

    # --- Create new Onset ---
    onset = STALTAOnset(position="classic", sampling_rate=50)
    onset.phases = ["P", "S"]
    onset.bandpass_filters = {
        "P": [1, 10, 4],
        "S": [1, 10, 4]}
    onset.sta_lta_windows = {
        "P": [0.1, 10.0],
        "S": [0.2, 10.0]}

    # --- Create new QuakeScan ---
    scan = QuakeScan(archive, lut, onset=onset, run_path=run_path,
                     run_name=run_name, log=True, loglevel="info")

    # --- Set detect parameters ---
    scan.timestep = 1.
    scan.threads = 4  # NOTE: increase as your system allows to increase speed!

    # --- Run detect ---
    scan.detect(starttime, endtime)


def generate_signal_trigger():
    """
    This script runs Trigger for the Iceland dike intrusion example.

    """

    # --- i/o paths ---
    lut_file = "./outputs/lut/wrangell.LUT"
    run_path = "./outputs/runs"

    # --- Set time period over which to run trigger ---
    starttime = "2016-09-26T15:40:00"
    endtime = "2016-09-26T15:50:00"
    run_name = f"{starttime[:13]}-{starttime[14:16]}_run"

    # --- Load the LUT ---
    lut = read_lut(lut_file=lut_file)

    # --- Create new Trigger ---
    trig = Trigger(lut, run_path=run_path, run_name=run_name, log=True,
                   loglevel="info")

    # --- Set trigger parameters ---
    # For a complete list of parameters and guidance on how to choose them, please
    # see the manual and read the docs.
    trig.marginal_window = 1.0
    trig.min_event_interval = 2.0
    trig.normalise_coalescence = True

    # # --- Static threshold ---
    # trig.threshold_method = "static"
    # trig.static_threshold = 1.45

    # --- Dynamic (Median Absolute Deviation) threshold ---
    trig.threshold_method = "dynamic"
    trig.mad_window_length = 1.
    trig.mad_multiplier = 5.

    # --- Toggle plotting options ---
    trig.plot_trigger_summary = True
    trig.xy_files = "./inputs/XY_FILES/wrangell_xyfiles.csv"

    # --- Run trigger ---
    # NOTE: here we use the optional "region" keyword argument to specify a spatial
    # filter for the triggered events. Only candidate events that fall within this
    # geographic area will be retained. This is useful for removing clear
    # artefacts; for example at the very edges of the grid.
    trig.trigger(starttime, endtime, interactive_plot=True)  # ,
    # region=[-17.15, 64.72, 0.0, -16.65, 64.93, 14.0])


def locate():
    """
    This script runs the locate stage for the Iceland dike intrusion example.

    """

    # --- i/o paths ---
    station_file = "./inputs/wrangell_stations.txt"
    response_file = "./inputs/resp.xml"
    data_in = "./inputs/mSEED"
    lut_file = "./outputs/lut/wrangell.LUT"
    run_path = "./outputs/runs"

    # --- Set time period over which to run locate ---
    starttime = "2016-09-26T15:40:00"
    endtime = "2016-09-26T15:50:00"
    run_name = f"{starttime[:13]}-{starttime[14:16]}_run"

    # --- Read in station file ---
    stations = read_stations(station_file)

    # --- Read in response inventory ---
    response_inv = read_response_inv(response_file)

    # --- Specify parameters for response removal ---
    response_params = AttribDict()
    # response_params.pre_filt = (0.05, 0.06, 30, 35)
    # response_params.water_level = 600

    # --- Create new Archive and set path structure ---
    archive = Archive(archive_path=data_in, stations=stations,
                      archive_format="YEAR/JD/STATION",
                      response_inv=response_inv,
                      response_removal_params=response_params)

    # --- Specify parameters for amplitude measurement ---
    amp_params = AttribDict()
    amp_params.signal_window = 5.0
    amp_params.highpass_filter = True
    amp_params.highpass_freq = 10.0

    # --- Specify parameters for magnitude calculation ---
    mag_params = AttribDict()
    mag_params.A0 = "Greenfield2018_bardarbunga"
    mag_params.amp_feature = "S_amp"

    mags = LocalMag(amp_params=amp_params, mag_params=mag_params,
                    plot_amplitudes=True)

    # --- Load the LUT ---
    lut = read_lut(lut_file=lut_file)

    # --- Create new Onset ---
    onset = STALTAOnset(position="centred", sampling_rate=50)
    onset.phases = ["P", "S"]
    onset.bandpass_filters = {
        "P": [1, 10, 4],
        "S": [1, 10, 4]}
    onset.sta_lta_windows = {
        "P": [0.1, 10.0],
        "S": [0.2, 10.0]}

    # --- Create new PhasePicker ---
    picker = GaussianPicker(onset=onset)
    picker.plot_picks = True

    # --- Create new QuakeScan ---
    scan = QuakeScan(archive, lut, onset=onset, picker=picker, mags=mags,
                     run_path=run_path, run_name=run_name, log=True,
                     loglevel="info")

    # --- Set locate parameters ---
    # For a complete list of parameters and guidance on how to choose them, please
    # see the manual and read the docs.
    scan.marginal_window = 1.0
    scan.threads = 4  # NOTE: increase as your system allows to increase speed!

    # --- Toggle plotting options ---
    scan.plot_event_summary = True
    scan.xy_files = "./inputs/XY_FILES/wrangell_xyfiles.csv"

    # --- Toggle writing of waveforms ---
    scan.write_cut_waveforms = True

    # --- Run locate ---
    scan.locate(starttime=starttime, endtime=endtime)

# FIXME: above functions are from QuakeMigrate notebooks and need to be tested
# TODO: check Quakemigrate notebook for other info


def picks_to_nonlinloc(marker_file_path):
    """ Reads the specified snuffler format marker file and converts it to
    NonLinLoc phase file format (written to present working directory):
    http://alomax.free.fr/nlloc/soft7.00/formats.html#_phase_nlloc_

    Returns: None
    Side effects: writes .obs file to current path

    Example:
        marker_file_path = "/Users/human/Dropbox/Programs/snuffler/loc_picks.mrkr"
        picks_to_nonlinloc(marker_file_path)
    """
    # read the marker file line by line
    with open("nll_picks.obs", "w") as write_file:
        with open(marker_file_path, 'r') as file:
            for index, line_Contents in enumerate(file):
                # process contents of line
                if len(line_Contents) > 52:  # avoid irrelevant short lines

                    # location picks file contains lines of uncertainties and
                    # lines of pick times. Pick times are ignored and the pick time
                    # is chosen as half of uncertainty range for simplicity
                    if (line_Contents[0:5] == 'phase') and (line_Contents[
                        -20:-19] == 'P') and (line_Contents[33:35] == '20') \
                            and (len(line_Contents) > 168):

                        pick_station = line_Contents[79:96].strip().split('.')[1]
                        # convert UGAP station names
                        if pick_station == "UGAP3":
                            pick_station = "103"
                        elif pick_station == "UGAP5":
                            pick_station = "105"
                        elif pick_station == "UGAP6":
                            pick_station = "106"

                        pick_channel = line_Contents[79:96].strip().split('.')[3]
                        start_time = UTCDateTime(line_Contents[7:32])
                        end_time = UTCDateTime(line_Contents[33:58])
                        one_sigma = (end_time - start_time) / 2
                        pick_time = start_time + one_sigma
                        first_motion = "?"

                        # write extracted information to nonlinloc phase file
                        line = f"{pick_station:<6} ?    N    ? P      ? " \
                               f"{pick_time.year}{pick_time.month:02}" \
                               f"{pick_time.day:02} {pick_time.hour:02}" \
                               f"{pick_time.minute:02} " \
                               f"{pick_time.second:02}." \
                               f"{int(str(round(pick_time.microsecond / 1000000, 4))[2:]):<04} GAU {one_sigma:1.2e}  0.00e+00  0.00e+00  0.00e+00    1.0000\n"
                        write_file.write(line)

                # # add blank lines between events # FIXME:
                # elif (line_Contents[0:5] == 'event') and (index != 1):
                #     line = "\n"
                #     write_file.write(line)

    return None


def process_nll_hypocenters(file_path):
    """ Reads a NonLinLoc .hyp file and writes the hypocenter locations to a
    file for plotting via GMT.

    Example:
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.3-0.45/loc/RR.sum.grid0.loc.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.4-0.55/loc/RR.sum.grid0.loc.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.5-0.65/loc/RR.sum.grid0.loc.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.6-0.75/loc/RR.sum.grid0.loc.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75/loc/RR.sum.grid0.loc.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.5-0.65/loc/RR.sum.grid0.loc.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.4-0.55/loc/RR.sum.grid0.loc.hyp"
        process_nll_hypocenters(file_path)
    """
    velocity_range = file_path[-33:-25]

    hypocenters = []
    # read the hypocenter file line by line
    with open(file_path, 'r') as file:
        SAVE_FLAG = False

        # process contents of each line
        for index, line_contents in enumerate(file):
            # only consider lines with accepted locations
            if line_contents[:5] == "NLLOC" and line_contents[
                                       -21:-3] == "Location completed":
                SAVE_FLAG = True

            # if line contains an accepted hypocenter: save it
            elif line_contents[:10] == "HYPOCENTER" and SAVE_FLAG:
                line = line_contents.split()
                hypocenter = [float(line[2]), float(line[4]),
                              round(float(line[6]) * 1000, 4)]

            # add uncertainty information
            elif line_contents[:21] == "QML_OriginUncertainty" and SAVE_FLAG:
                line = line_contents.split()
                hypocenter.append(round(float(line[6]) * 1000, 4))

                # set the save flag to False after each event
            elif line_contents[:9] == "END_NLLOC":
                if SAVE_FLAG:
                    hypocenters.append(hypocenter)
                SAVE_FLAG = False

    # generate files for plotting hypocenters via gmt
    with open(f"x_y_horizUncert_{velocity_range}.csv", "w") as write_file:
        # write header
        write_file.write("LON LAT Z\n")

        uncertys = np.asarray([hypocenter[3] for hypocenter in hypocenters])
        uncerty_min = uncertys.min()
        uncerty_max = uncertys.max()
        uncerty_range = uncerty_max - uncerty_min
        new_min = 1.3
        new_max = 9.0
        new_range = new_max - new_min

        # write each hypocenter to file
        for hypocenter in hypocenters:
            lat, lon = utm.to_latlon(hypocenter[0] * 1000,
                                      hypocenter[1] * 1000, 10, 'N')

            # scale horizontal uncertainty for plotting
            transformed_uncerty = ((hypocenter[3] - uncerty_min) * new_range \
                                  / uncerty_range) + new_min
            line = f"{lon} {lat} {transformed_uncerty}\n"
            write_file.write(line)

    with open(f"x_z_{velocity_range}.csv", "w") as write_file:
        # write header
        write_file.write("LON Z\n")

        # write each hypocenter to file
        for hypocenter in hypocenters:
            lat, lon = utm.to_latlon(hypocenter[0] * 1000,
                                      hypocenter[1] * 1000, 10, 'N')

            line = f"{lon} {hypocenter[2] * -1}\n"
            write_file.write(line)

    # inspect max and min elevations
    # a=np.asarray([hypocenter[2] for hypocenter in hypocenters])
    # a.min()
    # a.max()

    return None


def extract_nll_locations(file_path):
    """ Processes a .hyp file output from NonLinLoc's LocSum function
    containing PDF points for each event.

    Example:
        # specify the path to the summed location file (containing SCATTER lines)
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75/relocated/RR.hyp"
        extract_nll_locations(file_path)

    """
    hypocenters = []
    pdfs = []

    # read the hypocenter file line by line
    with open(file_path, 'r') as file:
        SAVE_FLAG = False
        SCATTER_FLAG = False

        # process contents of each line
        for index, line_contents in enumerate(file):

            # only consider lines with accepted locations
            if line_contents[:5] == "NLLOC" and line_contents[
                                                -21:-3] == "Location completed":
                SAVE_FLAG = True

            # if line contains an accepted hypocenter: save it
            elif line_contents[:10] == "HYPOCENTER" and SAVE_FLAG:
                line = line_contents.split()
                hypocenter = [float(line[2]), float(line[4]),
                              round(float(line[6]) * 1000, 4)]

            # if line contains RMS: save it
            elif line_contents[:7] == "QUALITY" and SAVE_FLAG:
                line = line_contents.split()
                hypocenter.append(float(line[8]))

            # check for pdf flag
            elif line_contents[:7] == "SCATTER":
                SCATTER_FLAG = True

            # add pdfs
            elif line_contents[0] == " " and SCATTER_FLAG and SAVE_FLAG:
                line = line_contents.split()
                pdfs.append([float(line[0]), float(line[1]), round(float(line[
                           2]) * 1000, 4), float(line[3])])

                # set the save flag to False after each event
            elif line_contents[:9] == "END_NLLOC":
                if SAVE_FLAG and SCATTER_FLAG:
                    hypocenters.append(hypocenter)
                SAVE_FLAG = False
                SCATTER_FLAG = False

    summed = 0
    for pdf in pdfs:
        summed += pdf[3]


    return None
