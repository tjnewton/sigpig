"""
Functions to assist with constraining the origin location of signals.
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
from obspy import read, UTCDateTime, Inventory, Stream
from obspy.clients.fdsn import Client
import glob
from obspy.core import AttribDict
from pyproj import Proj
import pandas as pd
from scipy.io import savemat
import numpy as np
from lidar import grids_from_raster
import netCDF4 as nc
import datetime
import matplotlib.pyplot as plt
import math
from data import rattlesnake_Ridge_Station_Locations
# from quakemigrate import QuakeScan, Trigger
# from quakemigrate.io import Archive, read_lut, read_stations, read_vmodel, \
#                             read_response_inv
# from quakemigrate.signal.onsets import STALTAOnset
# from quakemigrate.lut import compute_traveltimes
# from quakemigrate.signal.pickers import GaussianPicker
# from quakemigrate.signal.local_mag import LocalMag

# def download_date():
#     """
#     This script will download the waveform data and an instrument response
#     inventory from IRIS (in miniSEED and STATIONXML formats, respectively)
#     for the Iceland dike intrusion example.
#
#     """
#
#     # --- i/o paths ---
#     station_file = "./inputs/wrangell_stations.txt"
#     data_path = pathlib.Path("./inputs/mSEED")
#     response_file = "./inputs/resp.xml"
#
#     # --- Set network code & client ---
#     networks = ['AK', 'AT', 'AV', 'NP', 'TA', 'YG']
#     network_stations = {"AK": ["BAL", "BARN", "DHY", "DIV", "DOT", "GHO", "GLB",
#                                "K218", "KLU", "KNK", "MCAR", "MCK", "PAX", "PTPK",
#                                "RIDG", "RND", "SAW", "SCM", "VRDI", "WAT1",
#                                "WAT6", "WAT7"],
#                         "AT": ["MENT", "PMR"],
#                         "AV": ["WACK", "WASW", "WAZA"],
#                         "NP": ["2730", "2738", "2784", "8034", "AJKS", "AMJG"],
#                         "TA": ["HARP", "K24K", "L26K", "L27K", "M23K", "M24K",
#                                "M26K", "M27K", "N25K"],
#                         "YG": ["DEN1", "DEN3", "DEN4", "DEN5", "GLN1", "GLN2",
#                                "GLN3", "GLN4", "LKLO", "MCR1", "MCR2", "MCR3",
#                                "MCR4", "NEB1", "NEB2", "NEB3", "RH01", "RH02",
#                                "RH03", "RH04", "RH05", "RH06", "RH07", "RH08",
#                                "RH09", "RH10", "RH11", "RH12", "RH13", "RH14",
#                                "RH15", "TOK1", "TOK2", "TOK3", "TOK4", "TOK5"]}
#     network_channels = {"AK": ["BH*", "BN*", "HN*"], "AT": ["BH*"], "AV": [
#                         "BH*", "SH*"], "NP": ["HN*"], "TA": ["BH*"], "YG": ["BH*"]}
#
#     # --- Set time period ---
#     starttime = UTCDateTime("2016-09-26T15:40:00")
#     endtime = UTCDateTime("2016-09-26T15:50:00")
#
#     #  --- Read in station file ---
#     stations = read_stations(station_file)
#
#     # --- Download instrument response inventory ---
#     datacentre = "IRIS"
#     client = Client(datacentre)
#     inv = Inventory()
#     for network in networks:
#         for station in network_stations[network]:
#             try:
#                 inv += client.get_stations(network=network, station=station,
#                                            starttime=starttime, endtime=endtime,
#                                            level="response")
#             except Exception:
#                 pass
#     inv.write(response_file, format="STATIONXML")
#
#     # --- Make directories to store waveform data ---
#     waveform_path = data_path / str(starttime.year) / f"{starttime.julday:03d}"
#     waveform_path.mkdir(parents=True, exist_ok=True)
#
#     # --- Save waveform data subset ---
#     for network in networks:
#         for station in network_stations[network]:
#             # # load stream from daily waveform file
#             # file_list = glob.glob(f"../build_templates/data/{network}."
#             #                       f"{station}.***.{starttime.year}-"
#             #                       f"{starttime.month:02}-{starttime.day:02}.ms")
#             # # only consider first entry if multiple channels
#             # st = read(file_list[0])
#
#             for channel in network_channels[network]:
#                 try:
#                     print(f"Downloading waveform data for station {station} from {datacentre}")
#                     st = client.get_waveforms(network=network, station=station,
#                                               location="*", channel=channel,
#                                               starttime=starttime, endtime=endtime)
#
#                     st.merge(method=-1)
#                     for comp in ["E", "N", "Z"]:
#                         try:
#                             st_comp = st.select(component=comp)
#                             st_comp.write(str(waveform_path / f"{station}_{comp}.m"),
#                                           format="MSEED")
#                         except IndexError:
#                             pass
#
#                 except Exception:
#                     pass
#
#
# def generate_look_up_table():
#     """
#     This script generates the traveltime look-up table (LUT) for the Iceland dike
#     intrusion example.
#
#     """
#     # --- i/o paths ---
#     station_file = "./inputs/wrangell_stations.txt"
#     vmodel_file = "./inputs/wrangell_vmodel.txt"
#     lut_out = "./outputs/lut/wrangell.LUT"
#
#     # --- Read in the station information file ---
#     stations = read_stations(station_file)
#
#     # --- Read in the velocity model file ---
#     vmodel = read_vmodel(vmodel_file)
#
#     # --- Define the input and grid projections ---
#     gproj = Proj(proj="lcc", units="km", lon_0=-145.18, lat_0=62.14,
#                  lat_1=61.21,
#                  lat_2=63.07, datum="WGS84", ellps="WGS84", no_defs=True)
#     cproj = Proj(proj="longlat", datum="WGS84", ellps="WGS84", no_defs=True)
#
#     # --- Define the grid specifications ---
#     # AttribDict behaves like a Python dict, but also has '.'-style access.
#     grid_spec = AttribDict()
#     grid_spec.ll_corner = [-149.3515, 60.2834, -2.6]
#     grid_spec.ur_corner = [-141.0205, 663.9980, 100.0]
#     grid_spec.node_spacing = [1.0, 1.0, 0.5]
#     grid_spec.grid_proj = gproj
#     grid_spec.coord_proj = cproj
#
#     # --- 1-D velocity model LUT generation (using NonLinLoc eikonal solver) ---
#     lut = compute_traveltimes(grid_spec, stations, method="1dnlloc",
#                               vmod=vmodel,
#                               phases=["P", "S"], log=True, save_file=lut_out)
#
#
# def detect_signals():
#     """
#     This script runs the detect stage for the Iceland dike intrusion example.
#
#     """
#
#     # --- i/o paths ---
#     station_file = "./inputs/wrangell_stations.txt"
#     data_in = "./inputs/mSEED"
#     lut_file = "./outputs/lut/wrangell.LUT"
#     run_path = "./outputs/runs"
#
#     # --- Set time period over which to run detect ---
#     starttime = "2016-09-26T15:40:00"
#     endtime = "2016-09-26T15:50:00"
#     run_name = f"{starttime[:13]}-{starttime[14:16]}_run"
#
#     # --- Read in station file ---
#     stations = read_stations(station_file)
#
#     # --- Create new Archive and set path structure ---
#     archive = Archive(archive_path=data_in, stations=stations,
#                       archive_format="YEAR/JD/STATION")
#
#     # --- Load the LUT ---
#     lut = read_lut(lut_file=lut_file)
#     lut.decimate([2, 2, 2], inplace=True)
#
#     # --- Create new Onset ---
#     onset = STALTAOnset(position="classic", sampling_rate=50)
#     onset.phases = ["P", "S"]
#     onset.bandpass_filters = {
#         "P": [1, 10, 4],
#         "S": [1, 10, 4]}
#     onset.sta_lta_windows = {
#         "P": [0.1, 10.0],
#         "S": [0.2, 10.0]}
#
#     # --- Create new QuakeScan ---
#     scan = QuakeScan(archive, lut, onset=onset, run_path=run_path,
#                      run_name=run_name, log=True, loglevel="info")
#
#     # --- Set detect parameters ---
#     scan.timestep = 1.
#     scan.threads = 4  # NOTE: increase as your system allows to increase speed!
#
#     # --- Run detect ---
#     scan.detect(starttime, endtime)
#
#
# def generate_signal_trigger():
#     """
#     This script runs Trigger for the Iceland dike intrusion example.
#
#     """
#
#     # --- i/o paths ---
#     lut_file = "./outputs/lut/wrangell.LUT"
#     run_path = "./outputs/runs"
#
#     # --- Set time period over which to run trigger ---
#     starttime = "2016-09-26T15:40:00"
#     endtime = "2016-09-26T15:50:00"
#     run_name = f"{starttime[:13]}-{starttime[14:16]}_run"
#
#     # --- Load the LUT ---
#     lut = read_lut(lut_file=lut_file)
#
#     # --- Create new Trigger ---
#     trig = Trigger(lut, run_path=run_path, run_name=run_name, log=True,
#                    loglevel="info")
#
#     # --- Set trigger parameters ---
#     # For a complete list of parameters and guidance on how to choose them, please
#     # see the manual and read the docs.
#     trig.marginal_window = 1.0
#     trig.min_event_interval = 2.0
#     trig.normalise_coalescence = True
#
#     # # --- Static threshold ---
#     # trig.threshold_method = "static"
#     # trig.static_threshold = 1.45
#
#     # --- Dynamic (Median Absolute Deviation) threshold ---
#     trig.threshold_method = "dynamic"
#     trig.mad_window_length = 1.
#     trig.mad_multiplier = 5.
#
#     # --- Toggle plotting options ---
#     trig.plot_trigger_summary = True
#     trig.xy_files = "./inputs/XY_FILES/wrangell_xyfiles.csv"
#
#     # --- Run trigger ---
#     # NOTE: here we use the optional "region" keyword argument to specify a spatial
#     # filter for the triggered events. Only candidate events that fall within this
#     # geographic area will be retained. This is useful for removing clear
#     # artefacts; for example at the very edges of the grid.
#     trig.trigger(starttime, endtime, interactive_plot=True)  # ,
#     # region=[-17.15, 64.72, 0.0, -16.65, 64.93, 14.0])
#
#
# def locate():
#     """
#     This script runs the locate stage for the Iceland dike intrusion example.
#
#     """
#
#     # --- i/o paths ---
#     station_file = "./inputs/wrangell_stations.txt"
#     response_file = "./inputs/resp.xml"
#     data_in = "./inputs/mSEED"
#     lut_file = "./outputs/lut/wrangell.LUT"
#     run_path = "./outputs/runs"
#
#     # --- Set time period over which to run locate ---
#     starttime = "2016-09-26T15:40:00"
#     endtime = "2016-09-26T15:50:00"
#     run_name = f"{starttime[:13]}-{starttime[14:16]}_run"
#
#     # --- Read in station file ---
#     stations = read_stations(station_file)
#
#     # --- Read in response inventory ---
#     response_inv = read_response_inv(response_file)
#
#     # --- Specify parameters for response removal ---
#     response_params = AttribDict()
#     # response_params.pre_filt = (0.05, 0.06, 30, 35)
#     # response_params.water_level = 600
#
#     # --- Create new Archive and set path structure ---
#     archive = Archive(archive_path=data_in, stations=stations,
#                       archive_format="YEAR/JD/STATION",
#                       response_inv=response_inv,
#                       response_removal_params=response_params)
#
#     # --- Specify parameters for amplitude measurement ---
#     amp_params = AttribDict()
#     amp_params.signal_window = 5.0
#     amp_params.highpass_filter = True
#     amp_params.highpass_freq = 10.0
#
#     # --- Specify parameters for magnitude calculation ---
#     mag_params = AttribDict()
#     mag_params.A0 = "Greenfield2018_bardarbunga"
#     mag_params.amp_feature = "S_amp"
#
#     mags = LocalMag(amp_params=amp_params, mag_params=mag_params,
#                     plot_amplitudes=True)
#
#     # --- Load the LUT ---
#     lut = read_lut(lut_file=lut_file)
#
#     # --- Create new Onset ---
#     onset = STALTAOnset(position="centred", sampling_rate=50)
#     onset.phases = ["P", "S"]
#     onset.bandpass_filters = {
#         "P": [1, 10, 4],
#         "S": [1, 10, 4]}
#     onset.sta_lta_windows = {
#         "P": [0.1, 10.0],
#         "S": [0.2, 10.0]}
#
#     # --- Create new PhasePicker ---
#     picker = GaussianPicker(onset=onset)
#     picker.plot_picks = True
#
#     # --- Create new QuakeScan ---
#     scan = QuakeScan(archive, lut, onset=onset, picker=picker, mags=mags,
#                      run_path=run_path, run_name=run_name, log=True,
#                      loglevel="info")
#
#     # --- Set locate parameters ---
#     # For a complete list of parameters and guidance on how to choose them, please
#     # see the manual and read the docs.
#     scan.marginal_window = 1.0
#     scan.threads = 4  # NOTE: increase as your system allows to increase speed!
#
#     # --- Toggle plotting options ---
#     scan.plot_event_summary = True
#     scan.xy_files = "./inputs/XY_FILES/wrangell_xyfiles.csv"
#
#     # --- Toggle writing of waveforms ---
#     scan.write_cut_waveforms = True
#
#     # --- Run locate ---
#     scan.locate(starttime=starttime, endtime=endtime)

# FIXME: above functions are from QuakeMigrate notebooks and need to be tested
# TODO: check Quakemigrate notebook for other info


def picks_to_nonlinloc(marker_file_path, waveform_files_path):
    """ Reads the specified snuffler format marker file and converts it to
    NonLinLoc phase file format (written to present working directory):
    http://alomax.free.fr/nlloc/soft7.00/formats.html#_phase_nlloc_

    Returns: None
    Side effects: writes .obs file to current path

    Example:
        # define the path to the snuffler-format marker file containing picks
        marker_file_path = "event_picks.mrkr"
        # define the path to the directory containing waveform files
        waveform_files_path = "/Users/human/Desktop/RR_MSEED"

        picks_to_nonlinloc(marker_file_path, waveform_files_path)
    """
    # keep track of event ID's
    event_ids = []
    phases_dict = {}
    first_event = True
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
                        # get the event id (snuffler time hash)
                        event_id = line_Contents[-77:-49]
                        # check if event has been found yet
                        if event_id not in event_ids:
                            event_ids.append(event_id)
                            phases_dict[event_id] = []
                            # the first event header has location information
                            if first_event:
                                write_file.write("# EQEVENT:  Label: EQ001  "
                                                 "Loc:  X 10.0  Y 100.0  Z "
                                                 "0.07  OT 0.00\n")
                                first_event = False
                            else:
                                # subsequent event headers are generic
                                event_header = "#\n# EQEVENT:\n"
                                write_file.write(event_header)

                        # get station and channel of pick
                        pick_station = line_Contents[-95:-78].strip().split('.')[1]
                        pick_channel = line_Contents[-95:-78].strip().split('.')[3]

                        # check if this pick is a duplicate
                        sta_chan = f"{pick_station}.{pick_channel}"
                        if sta_chan not in phases_dict[event_id]:
                            # append the phase to keep track of duplicates
                            phases_dict[event_id].append(sta_chan)

                            start_time = UTCDateTime(line_Contents[7:32])
                            end_time = UTCDateTime(line_Contents[33:58])
                            # check if times need to be swapped for some reason
                            if start_time > end_time:
                                swap = start_time
                                start_time = end_time
                                end_time = swap

                            one_sigma = (end_time - start_time) / 2
                            pick_time = start_time + one_sigma
                            first_motion = "?"

                            # get amplitude information from the trace
                            try:
                                # first find all files for specified day
                                day_file_list = sorted(
                                    glob.glob(f"{waveform_files_path}/"
                                              f"*.{pick_station}.."
                                              f"{pick_channel}"
                                              f".{pick_time.year}-{pick_time.month:02}"
                                              f"-{pick_time.day:02}*.ms"))

                                # should only be one file, guard against many
                                file = day_file_list[0]

                                # load file into stream
                                st = Stream()
                                st += read(file)

                                st.trim(start_time - 1, end_time + 2, pad=True,
                                        fill_value=0, nearest_sample=True)
                                # interpolate to 250 Hz
                                st.interpolate(sampling_rate=250.0)
                                # detrend
                                # st[0].detrend("demean")
                                # bandpass filter
                                st.filter("bandpass", freqmin=20, freqmax=60,
                                          corners=4)
                                # trim trace surrounding pick time
                                st.trim(start_time - 0.01, pick_time + 0.1, pad=True,
                                        fill_value=0, nearest_sample=True)

                                # get the trace data
                                trace = st[0].data
                                # find the maximum and minimum value in the trace
                                max_idx = np.nanargmax(trace)
                                min_idx = np.nanargmin(trace)
                                # get the peak to peak amplitude and period
                                peak_to_peak_amplitude = trace[max_idx] - trace[
                                                                           min_idx]
                                peak_to_peak_period = abs(max_idx - min_idx) / 250

                            # if something failed, use zeros
                            except Exception:
                                peak_to_peak_amplitude = 0
                                peak_to_peak_period = 0
                                pass

                            peak_to_peak_amplitude = np.log(peak_to_peak_amplitude)

                            # convert UGAP station names
                            if pick_station == "UGAP3":
                                pick_station = "103"
                            elif pick_station == "UGAP5":
                                pick_station = "105"
                            elif pick_station == "UGAP6":
                                pick_station = "106"

                            # write extracted information to nonlinloc phase file
                            line = f"{pick_station:<6} ?    N    ? P      ? " \
                                   f"{pick_time.year}{pick_time.month:02}" \
                                   f"{pick_time.day:02} {pick_time.hour:02}" \
                                   f"{pick_time.minute:02} " \
                                   f"{pick_time.second:02}." \
                                   f"{int(str(round(pick_time.microsecond / 1000000, 4))[2:]):<04} GAU {one_sigma:1.2e}  0.00e+00  {peak_to_peak_amplitude:1.2e}  {peak_to_peak_period:1.2e}    {peak_to_peak_amplitude:1.2e}\n"
                            write_file.write(line)
                            # f"{int(str(round(pick_time.microsecond / 1000000, 4))[2:]):<04} GAU {one_sigma:1.2e}  0.00e+00  {peak_to_peak_amplitude:1.2e}  {peak_to_peak_period:1.2e}    1.0000\n"

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
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75_newPicks/loc/RR.sum.grid0.loc.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75_newPicks/relocated/RR.sum.grid0.loc.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75_214Picks/relocated/RR.sum.grid0.loc.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75_214Picks/loc/RR.sum.grid0.loc.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.5-0.65_214Picks/relocated/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.4-0.55_214Picks/relocated/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75_214Picks_OSOS_0.04/relocated/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75_214Picks_OSOS_0.1/relocated/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75_214Picks_OSOS_0.25/relocated/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75_214Picks_OSOS_0.5/relocated/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75_214Picks_OSOS_0.75/relocated/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75_214Picks_OSOS_0.5_amp/relocated/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75_214Picks_OSOS_0.04_amp/relocated/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75_214Picks_OSOS_0.5_amp_scale/relocated/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.6-0.75_214Picks_OSOS_0.5_amp_scale/loc/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.6-0.75_214Picks_OSOS_0.75_amp_scale/loc/RR.hyp"
        process_nll_hypocenters(file_path)

        # then plot with gmt/RR_events_plot.sh
    """
    velocity_range = file_path[-33:-25]
    velocity_range = "0.6-0.75"

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
        # get event pdfs and hypocenters via this function
        pdfs, hypocenters = extract_nll_locations(file_path)
    """
    # define containers to hold results
    hypocenters = []
    pdfs = []
    rms = []
    # velocity_range = file_path[-25:-17]
    velocity_range = "0.5-0.65"

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
            elif line_contents[:10] == "HYPOCENTER":
                line = line_contents.split()
                hypocenter = [float(line[2]), float(line[4]),
                              round(float(line[6]) * 1000, 4)]

            # if line contains RMS: save it
            elif line_contents[:7] == "QUALITY":
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

    # # generate file with all pdf points
    # with open(f"xyw_{velocity_range}.csv", "w") as write_file:
    #     # write header
    #     write_file.write("X Y W\n")
    #
    #     # write each hypocenter to file
    #     for pdf in pdfs:
    #         lat, lon = utm.to_latlon(pdf[0] * 1000, pdf[1] * 1000, 10, 'N')
    #
    #         line = f"{lon} {lat} {pdf[3]}\n"
    #         write_file.write(line)
    #
    # # generate file with all pdf points
    # with open(f"xzw_{velocity_range}.csv", "w") as write_file:
    #     # write header
    #     write_file.write("X Z W\n")
    #
    #     # write each hypocenter to file
    #     for pdf in pdfs:
    #         lat, lon = utm.to_latlon(pdf[0] * 1000, pdf[1] * 1000, 10, 'N')
    #
    #         line = f"{lon} {pdf[2] * -1} {pdf[3]}\n"
    #         write_file.write(line)

    pdfs = pd.DataFrame(pdfs, columns = ['x', 'y', 'z', 'weight'])
    hypocenters = np.asarray(hypocenters)
    print(hypocenters[:,3].sum())

    return pdfs, hypocenters


def location_pdfs_to_grid(pdfs, project_name):
    """ Takes in a list of lists, where each list entry contains:
        [x y z weight], and outputs 3 NetCDF grids (x-z, x-y, and y-z planes)
        containing the sum of weights in grid nodes to plot via GMT.

        Example:
            # specify the path to the summed location file (containing SCATTER lines)
            # file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75/relocated/RR.hyp"
            # file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.5-0.65/relocated/RR.hyp"
            # file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.4-0.55/relocated/RR.hyp"
            # file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.6-0.75/loc/RR.hyp"
            # file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.5-0.65/loc/RR.hyp"
            # file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.4-0.55/loc/RR.hyp"
            # file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.3-0.45/loc/RR.hyp"
            # file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75_newPicks/relocated/RR.hyp"
            # file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.5-0.65_newPicks/relocated/RR.hyp"
            # file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.4-0.55_newPicks/relocated/RR.hyp"
            # file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75_214Picks/relocated/RR.hyp"
            # file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.5-0.65_214Picks/relocated/RR.hyp"
            # file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.4-0.55_214Picks/relocated/RR.hyp"
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75_214Picks_OSOS_0.04/relocated/RR.hyp"
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75_214Picks_OSOS_0.1/relocated/RR.hyp"
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75_214Picks_OSOS_0.25/relocated/RR.hyp"
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75_214Picks_OSOS_0.5/relocated/RR.hyp"
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75_214Picks_OSOS_0.75/relocated/RR.hyp"
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75_214Picks_OSOS_0.5_amp/relocated/RR.hyp"
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75_214Picks_OSOS_0.04_amp/relocated/RR.hyp"
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_ssst-coh_rr_0.6-0.75_214Picks_OSOS_0.5_amp_scale/relocated/RR.hyp"
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.6-0.75_214Picks_OSOS_0.5_amp_scale/loc/RR.hyp"
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.6-0.75_214Picks_OSOS_0.75_amp_scale/loc/RR.hyp"
            pdfs, hypocenters = extract_nll_locations(file_path)
            project_name = "Rattlesnake Ridge"
            location_pdfs_to_grid(pdfs, project_name)

            # then plot with gmt/RR_pdfs_grid.sh
    """

    if project_name == "Rattlesnake Ridge":
        # load arrays from raster file
        raster_file = '/Users/human/Dropbox/Programs/lidar/yakima_basin_2018_dtm_43.tif'

        # expanded spatial limits for plotting NLL results in GMT
        x_limits = [694.10, 694.50]
        y_limits = [5155.3, 5155.99]
        z_limits = [-150, 9]

        # get x and y distance in meters
        x_dist_m = (x_limits[1] - x_limits[0]) * 1000
        y_dist_m = (y_limits[1] - y_limits[0]) * 1000
        z_dist_m = (z_limits[1] - z_limits[0])
        # x and y steps for loops
        num_x_steps = int(x_dist_m / 3)  # 3 m resolution
        num_y_steps = int(y_dist_m / 3)
        num_z_steps = int(z_dist_m / 3)
        x_step = round((x_limits[1] - x_limits[0]) / num_x_steps, 3)
        y_step = round((y_limits[1] - y_limits[0]) / num_y_steps, 3)
        z_step = round((z_limits[1] - z_limits[0]) / num_z_steps, 3)

        # initialize grid lists for x_y plot
        longitude_grid = []
        latitude_grid = []
        weights_grid = []

        # build x_y plot grid structures:
        # loop over rows from top to bottom and sample at dataset resolution
        for row in range(num_y_steps):
            latitude = y_limits[1] - (row * y_step)
            row_latitudes = []
            row_longitudes = []
            row_weights = []
            # loop over each column and build grid
            for column in range(num_x_steps):
                weight_sum = 0
                longitude = x_limits[0] + (column * x_step)
                row_latitudes.append(latitude)
                row_longitudes.append(longitude)

                # find entries in pdfs that are in this grid cell
                grid_cell_pdfs = pdfs.loc[(pdfs['x'] >= longitude) & (pdfs['x']
                                        < longitude + x_step) & (pdfs['y']
                                        <= latitude) & (pdfs['y'] > latitude
                                        - y_step)]
                # if there are weights in this cell, sum them
                if len(grid_cell_pdfs) > 0:
                    row_weights.append(grid_cell_pdfs['weight'].sum())

                # if no weights are in this cell, return 0
                else:
                    row_weights.append(0)


            # convert utm to lats/lons for entire row
            lats_lons = utm.to_latlon(np.asarray(row_longitudes) * 1000,
                                      np.asarray(row_latitudes) * 1000, 10,
                                      'N')

            # convert eastings and northings to lat/lon
            row_lons = [lon for lon in lats_lons[1]]
            row_lats = [lat for lat in lats_lons[0]]
            longitude_grid.append(row_lons)
            latitude_grid.append(row_lats)
            weights_grid.append(row_weights)

        longitude_grid = np.asarray(longitude_grid)
        latitude_grid = np.asarray(latitude_grid)
        weights_grid = np.asarray(weights_grid)

        # initialize grid lists for x_z plot
        x_grid = []
        z_grid = []
        xz_weights_grid = []
        # changes z values to negative
        pdfs['z'] *= -1

        # build x_z plot grid structures:
        # loop over rows from top to bottom and sample at dataset resolution
        for row in range(num_z_steps):
            z = z_limits[1] - (row * z_step)
            row_zs = []
            row_latitudes = []
            row_longitudes = []
            row_weights = []
            # loop over each column and build grid
            for column in range(num_x_steps):
                weight_sum = 0
                longitude = x_limits[0] + (column * x_step)
                row_zs.append(z)
                row_longitudes.append(longitude)
                # add dummy latitude for conversion
                row_latitudes.append(((y_limits[1] - y_limits[0]) / 2) +
                                     y_limits[0])

                # find entries in pdfs that are in this grid cell
                grid_cell_pdfs = pdfs.loc[(pdfs['x'] >= longitude) & (pdfs['x']
                                          < longitude + x_step) & (pdfs['z']
                                          <= z) & (pdfs['z'] > z - z_step)]
                # if there are weights in this cell, sum them
                if len(grid_cell_pdfs) > 0:
                    row_weights.append(grid_cell_pdfs['weight'].sum())

                # if no weights are in this cell, return 0
                else:
                    row_weights.append(0)

            # convert utm to lats/lons for entire row
            lats_lons = utm.to_latlon(np.asarray(row_longitudes) * 1000,
                                      np.asarray(row_latitudes) * 1000, 10,
                                      'N')

            # convert eastings and northings to lat/lon
            row_lons = [lon for lon in lats_lons[1]]
            row_lats = [lat for lat in lats_lons[0]]
            x_grid.append(row_lons)
            z_grid.append(row_zs)
            xz_weights_grid.append(row_weights)

        x_grid = np.asarray(x_grid)
        z_grid = np.asarray(z_grid)
        xz_weights_grid = np.asarray(xz_weights_grid)

        # initialize grid lists for y_z plot
        zz_grid = []
        yy_grid = []
        yz_weights_grid = []

        # build y_z plot grid structures:
        # loop over rows from top to bottom and sample at dataset resolution
        for row in range(num_y_steps):
            latitude = y_limits[1] - (row * y_step)
            row_zs = []
            row_latitudes = []
            row_longitudes = []
            row_weights = []
            # loop over each column and build grid
            for column in range(num_z_steps):
                weight_sum = 0
                z = z_limits[0] + (column * z_step)
                row_zs.append(z)
                row_latitudes.append(latitude)
                # add dummy longitude for conversion
                row_longitudes.append(((x_limits[1] - x_limits[0]) / 2) +
                                      x_limits[0])

                # find entries in pdfs that are in this grid cell
                grid_cell_pdfs = pdfs.loc[(pdfs['y'] >= latitude) & (pdfs['y']
                                          < latitude + y_step) & (pdfs['z']
                                          <= z) & (pdfs['z'] > z - z_step)]
                # if there are weights in this cell, sum them
                if len(grid_cell_pdfs) > 0:
                    row_weights.append(grid_cell_pdfs['weight'].sum())

                # if no weights are in this cell, return 0
                else:
                    row_weights.append(0)

            # convert utm to lats/lons for entire row
            lats_lons = utm.to_latlon(np.asarray(row_longitudes) * 1000,
                                      np.asarray(row_latitudes) * 1000, 10,
                                      'N')

            # convert eastings and northings to lat/lon
            row_lons = [lon for lon in lats_lons[1]]
            row_lats = [lat for lat in lats_lons[0]]
            zz_grid.append(row_zs)
            yy_grid.append(row_lats)
            yz_weights_grid.append(row_weights)

        zz_grid = np.asarray(zz_grid)
        yy_grid = np.asarray(yy_grid)
        yz_weights_grid = np.asarray(yz_weights_grid)

        # save x_y_weights grid to NetCDF file
        filename = 'gridded_rr_pdfs.nc'
        ds = nc.Dataset(filename, 'w', format='NETCDF4')
        # add no time dimension, and lat & lon dimensions
        time = ds.createDimension('time', None)
        lat = ds.createDimension('lat', num_y_steps)
        lon = ds.createDimension('lon', num_x_steps)
        # generate netCDF variables to store data
        times = ds.createVariable('time', 'f4', ('time',))
        lats = ds.createVariable('lat', 'f4', ('lat',))
        lons = ds.createVariable('lon', 'f4', ('lon',))
        value = ds.createVariable('value', 'f4', ('time', 'lat', 'lon',))
        value.units = 'm'
        # set spatial values of grid
        lats[:] = np.flip(latitude_grid[:,0])
        lons[:] = longitude_grid[0,:]
        value[0, :, :] = np.flip(weights_grid, axis=0)
        # close the netCDF file
        ds.close()
        print(f"Max weight sum: {weights_grid.max()}")

        # save x_z_weights grid to NetCDF file
        filename = 'gridded_rr_pdfs_xz.nc'
        ds = nc.Dataset(filename, 'w', format='NETCDF4')
        # add no time dimension, and lat & lon dimensions
        time = ds.createDimension('time', None)
        lat = ds.createDimension('lat', num_z_steps)
        lon = ds.createDimension('lon', num_x_steps)
        # generate netCDF variables to store data
        times = ds.createVariable('time', 'f4', ('time',))
        lats = ds.createVariable('lat', 'f4', ('lat',))
        lons = ds.createVariable('lon', 'f4', ('lon',))
        value = ds.createVariable('value', 'f4', ('time', 'lat', 'lon',))
        value.units = 'm'
        # set spatial values of grid
        lats[:] = np.flip(z_grid[:, 0])
        lons[:] = x_grid[0, :]
        value[0, :, :] = np.flip(xz_weights_grid, axis=0)
        # close the netCDF file
        ds.close()
        print(f"Max weight sum: {xz_weights_grid.max()}")

        # save y_z_weights grid to NetCDF file
        filename = 'gridded_rr_pdfs_yz.nc'
        ds = nc.Dataset(filename, 'w', format='NETCDF4')
        # add no time dimension, and lat & lon dimensions
        time = ds.createDimension('time', None)
        lat = ds.createDimension('lat', num_y_steps)
        lon = ds.createDimension('lon', num_z_steps)
        # generate netCDF variables to store data
        times = ds.createVariable('time', 'f4', ('time',))
        lats = ds.createVariable('lat', 'f4', ('lat',))
        lons = ds.createVariable('lon', 'f4', ('lon',))
        value = ds.createVariable('value', 'f4', ('time', 'lat', 'lon',))
        value.units = 'm'
        # set spatial values of grid
        lats[:] = np.flip(yy_grid[:, 0])
        # hack to plot depth reversed in GMT
        zz_grid = zz_grid * -1
        lons[:] = np.flip(zz_grid[0, :])
        value[0, :, :] = np.rot90(np.rot90(yz_weights_grid))
        # close the netCDF file
        ds.close()
        print(f"Max weight sum: {yz_weights_grid.max()}")

    return None


def find_max_hypocenter(hypocenters):
    """Finds the index of the hypocenter that is closest to the upper right
    corner (themax coordinates) of the project location grid. """
    x_limits = [694.10, 694.50]
    y_limits = [5155.3, 5155.99]

    # find the distance from each hypocenter to the upper right grid corner
    distances = []
    for hypocenter in hypocenters:
        distances.append(np.sqrt((hypocenter[0] - x_limits[1])**2 + (
                         hypocenter[1] - y_limits[1])**2))

    distances = np.asarray(distances)
    index = distances.argmin()
    lats_lons = utm.to_latlon(np.asarray(hypocenters[index,0]) * 1000,
                              np.asarray(hypocenters[index,1]) * 1000, 10,
                              'N')

    return