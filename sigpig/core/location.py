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
from obspy import read, UTCDateTime, Inventory, Stream, read_inventory
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
import pickle
import utm
from data import rattlesnake_Ridge_Station_Locations
from lidar import elevations_from_raster
import scipy.io as sio
from matplotlib import cm
from obspy.core import Stream, Trace
from tqdm import tqdm
import time


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
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.6-0.75_214Picks_OSOS_0.25_amp_scale/loc/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.6-0.75_214Picks_OSOS_0.1_amp_scale/loc/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.6-0.75_214Picks_OSOS_0.04_amp_scale/loc/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.6-0.75_214Picks_amp_scale/loc/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.5-0.65_214Picks_amp_scale/loc/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.4-0.55_214Picks_amp_scale/loc/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.3-0.45_214Picks_amp_scale/loc/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.6-0.75_214Picks_amp_scale_ssst/relocated/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.5-0.65_214Picks_amp_scale_ssst/relocated/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.4-0.55_214Picks_amp_scale_ssst/relocated/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.3-0.45_214Picks_amp_scale_ssst/relocated/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.6-0.75_214Picks_amp_scale_syn/loc/RR.hyp"
        file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.6-0.75_214Picks_amp_scale_ssst_syn/relocated/RR.hyp"
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

    # access file with rejected events for RMS summing
    rms_file_path = file_path[:-3] + 'sum.grid0.loc.hyp'

    # get the summed RMS among all accepted and rejected events
    with open(rms_file_path, 'r') as file:
        # process contents of each line
        for index, line_contents in enumerate(file):
            # if line contains RMS: save it
            if line_contents[:7] == "QUALITY":
                line = line_contents.split()
                rms.append(float(line[8]))

    # store accepted hypocenters
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
    rms = np.asarray(rms)
    print(f"RMS sum: {rms.sum()}")

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
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.6-0.75_214Picks_OSOS_0.25_amp_scale/loc/RR.hyp"
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.6-0.75_214Picks_OSOS_0.1_amp_scale/loc/RR.hyp"
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.6-0.75_214Picks_OSOS_0.04_amp_scale/loc/RR.hyp"
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.6-0.75_214Picks_amp_scale/loc/RR.hyp"
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.5-0.65_214Picks_amp_scale/loc/RR.hyp"
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.4-0.55_214Picks_amp_scale/loc/RR.hyp"
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.3-0.45_214Picks_amp_scale/loc/RR.hyp"
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.6-0.75_214Picks_amp_scale_ssst/relocated/RR.hyp"
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.5-0.65_214Picks_amp_scale_ssst/relocated/RR.hyp"
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.4-0.55_214Picks_amp_scale_ssst/relocated/RR.hyp"
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.3-0.45_214Picks_amp_scale_ssst/relocated/RR.hyp"
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.6-0.75_214Picks_amp_scale_syn/loc/RR.hyp"
            file_path = "/Users/human/Dropbox/Research/Rattlesnake_Ridge/nlloc_rr_0.6-0.75_214Picks_amp_scale_ssst_syn/relocated/RR.hyp"
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

        # # helper for converting synthetics
        # row_longitudes = [694.337]
        # row_latitudes = [5155.76]
        # lats_lons = utm.to_latlon(np.asarray(row_longitudes) * 1000,
        #                           np.asarray(row_latitudes) * 1000, 10,
        #                           'N')
        # row_lons = [lon for lon in lats_lons[1]]
        # row_lats = [lat for lat in lats_lons[0]]
        # print(f"{row_lons[0]} {row_lats[0]}")

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


def generate_amplitude_station_file():
    """Generates a file of station information in the format necessary for
    amplidue-based locations programs.

    Example:
        generate_amplitude_station_file()
    """

    dfdict = {'name': [], 'northing': [], 'easting': [],
              'elevation': [], 'latitude': [], 'longitude': []}

    # get station locations on specified date from native GPS elevation
    date = UTCDateTime("2018-03-13T00:04:00.0Z")
    station_locations = rattlesnake_Ridge_Station_Locations(date)

    # get station elevations from DEM rather than using native GPS
    # elevations
    raster_file = '/Users/human/Dropbox/Programs/lidar/yakima_basin_2018_dtm_43.tif'
    for station in station_locations.keys():
        elevation = elevations_from_raster(raster_file,
                                       [station_locations[station][1]],
                                       [station_locations[station][0]])
        station_locations[station][2] = elevation[0]

    # assemble dict in proper format for Stingray
    for station in station_locations.keys():
        easting, northing, _, _ = utm.from_latlon(station_locations[
                                                  station][0],
                                                  station_locations[
                                                  station][1])

        # append eastings, northings, and elevations in m
        dfdict['northing'].append(northing)
        dfdict['easting'].append(easting)
        dfdict['latitude'].append(station_locations[station][0])
        dfdict['longitude'].append(station_locations[station][1])
        dfdict['elevation'].append(station_locations[station][2])

        # convert UGAP station names
        if station == "UGAP3":
            station = "103"
        elif station == "UGAP5":
            station = "105"
        elif station == "UGAP6":
            station = "106"
        dfdict['name'].append([station])

    # write station locations to file
    with open('amplitude_station_locations.dat', 'a') as file:
        for index in range(len(dfdict['name'])):
            # write station, easting, northing, elevation
            file.write(f'{dfdict["name"][index][0]} '
                       f'{dfdict["latitude"][index]},'
                       f'{dfdict["longitude"][index]},'
                       f'{dfdict["northing"][index]},'
                       f'{dfdict["easting"][index]},'
                       f'{dfdict["elevation"][index]}\n')

    return None


def set_params():
    beta=400 # m/s
    f=60 # frequency
    Q=50 # quality factor (??)
    return beta, f, Q


def locate(refstaid,staids,statts):
    """
    refstaid - reference station
    staids - station ids
    statts - difference between arrival at station and reference station
    """
    # calculate travel time difference relative to reference station
    ind=np.where(staids==str(refstaid))[0][0]
    stadts=statts-statts[ind]
    file='/Users/human/Dropbox/Research/Rattlesnake_Ridge/amplitude_locations/ray_tracing/tts_'+str(refstaid)+'.pkl'
    #print(file)
    with open(file, 'rb') as f:
        reftts = pickle.load(f)
    residuals=np.zeros_like(reftts)
    for ii in range(len(staids)):
        #print(ii)
        if refstaid != staids[ii]:
            file='/Users/human/Dropbox/Research/Rattlesnake_Ridge' \
                 '/amplitude_locations/ray_tracing/tts_'+str(staids[ii])+'.pkl'
            with open(file, 'rb') as f:
                tts = pickle.load(f)
            pred=tts-reftts # model difference between ref and arrival
            dts=np.abs(stadts[ii]-pred) # difference between observed and predicted differences
            residuals+=dts
    return residuals


def amplocate(staids,ampfacs,staamps,de,dn,dz,A0,dists, weight=2,
              dijkstra=1):
    """
    Locates events based on amplitude
    Inputs:
        staids - station ids that correspond to amplitudes in staamps
        ampfacs - amplitude factors for each station in staids
        staamps - amplitudes measured at each station (set to zero if no pick), needs to be in same order as staids
        A0 - initial amplitudes to consider
    Outputs:
        residuals - sum of squared differences between normalized predicted amps and normalized actual amps
    """
    # some constants
    beta, f, Q = set_params()
    B=np.pi*f/(Q*beta) # 1/m
    # add amplification factor correction
    staamps=staamps/ampfacs
    if weight==0:
        weights=np.ones(len(staamps))
    elif weight==1:
        weights=((np.max(dists)-dists)/np.max(dists))
    elif weight==2:
        weights=((np.max(dists)-dists)/np.max(dists))**2
    # initialize array for residuals
    residuals=np.zeros((len(de),len(dn),len(dz),len(A0)))
    for ii in range(len(staids)):
        if dijkstra==1:
            file='/Users/human/Dropbox/Research/Rattlesnake_Ridge/amplitude_locations/ray_tracing/dists_'+str(staids[ii])+'.pkl'
            # file='/Users/human/Dropbox/Research/Rattlesnake_Ridge' \
            #      '/amplitude_locations/ray_tracing' \
            #      '/amandas_old_files/dijkstra_dists_'+str(staids[ii])+'.pkl'

        else:
            file='/Users/human/Dropbox/Research/Rattlesnake_Ridge/amplitude_locations/ray_tracing/dists_'+str(staids[ii])+'.pkl'
        with open(file, 'rb') as f:
            r = pickle.load(f)
        for jj in range(len(A0)):
            amp=A0[jj]*np.exp(-B*r)/r
            damps=(staamps[ii]-amp)**2*weights[ii] # difference between observed and predicted differences
            residuals[:,:,:,jj] = residuals[:,:,:,jj]+damps
    minind = np.unravel_index(np.argmin(residuals, axis=None), residuals.shape)
    predamps=np.zeros(len(staids))
    for ii in range(len(staids)):
        # file='/Users/human/Dropbox/Research/Rattlesnake_Ridge' \
        #      '/amplitude_locations/ray_tracing' \
        #      '/amandas_old_files/dists_'+str(staids[ii])+'.pkl'
        file='/Users/human/Dropbox/Research/Rattlesnake_Ridge/amplitude_locations/ray_tracing/dists_'+str(staids[ii])+'.pkl'
        with open(file, 'rb') as f:
            r = pickle.load(f)
        r=r[minind[0],minind[1],minind[2]]
        predamps[ii]=A0[minind[3]]*np.exp(-B*r)/r
    err=100*np.sqrt(np.sum((staamps-predamps)**2)/np.sum(staamps**2))
    return residuals, staamps, predamps, err, weights


def fakequake(x,y,z,A0,staids):
    """
    Calculates travel times for a fake earthquake
    Inputs:
        x - x loc of quake
        y - y loc of quake
        z - z loc of quake
        staids - station ids
    Outputs:
        dts - predicted travel times from source to all stations
        damps - predicted amplitudes at all stations
    """
    # some constants
    beta, f, Q = set_params()
    B=np.pi*f/(Q*beta) # 1/m
    # get coordinates
    de, dn, dz, _=get_coords()
    xind=np.where(de==x)[0][0]
    yind=np.where(dn==y)[0][0]
    zind=np.where(dz==z)[0][0]
    dts=np.zeros(len(staids))
    damps=np.zeros(len(staids))
    for ii in range(len(staids)):
        file='/Users/human/Dropbox/Research/Rattlesnake_Ridge/amplitude_locations/ray_tracing/tts_'+str(staids[ii])+'.pkl'
        with open(file, 'rb') as f:
            tts = pickle.load(f)
        dts[ii]=tts[xind,yind,zind]
        file='/Users/human/Dropbox/Research/Rattlesnake_Ridge/amplitude_locations/ray_tracing/dists_'+str(staids[ii])+'.pkl'
        with open(file, 'rb') as f:
            dists = pickle.load(f)
        r=dists[xind,yind,zind]
        damps[ii]=A0*np.exp(-B*r)/r
    return dts, damps


def plot_resid(X,Y,ampresid,minind,mininds,de,dn,dz,locs,staids,A0s):
    fig,axs=plt.subplots(nrows=1,ncols=1,figsize=(6,8))
    # plot solution
    cs=axs.contourf(X, Y, np.log10(ampresid[:,:,minind[2],minind[3]].T), 30, cmap=cm.PuBu_r)
    axs.axis('equal')
    for ii in range(len(mininds[0])):
        axs.plot(de[mininds[0][ii]],dn[mininds[1][ii]],'ko') # plot location
    axs.plot(de[minind[0]],dn[minind[1]],'ro') # plot location
    axs.plot(locs['utmx'].values,locs['utmy'].values,'r^') # plot all the stations
    for kk in range(len(locs)): # plot all station names
        axs.text(locs['utmx'].values[kk],locs['utmy'].values[kk],str(locs.iloc[kk]['station']))
    axs.set_title('Depth: '+str(dz[minind[2]])+' m Amp: '+str(A0s[minind[3]]))
    axs.set_xlim((np.min(X),np.max(X)))
    axs.set_ylim((np.min(Y),np.max(Y)))
    fig.colorbar(cs)
    return None


def plot_resid_slices(X,Y,ampresid,minind,de,dn,dz,locs,staids,A0s):
    for ii in np.arange(0,40,5):
        fig,axs=plt.subplots(nrows=1,ncols=1,figsize=(6,8))
        # plot solution
        cs=axs.contourf(X, Y, np.log10(ampresid[:,:,ii,minind[3]].T), 30, cmap=cm.PuBu_r)
        axs.axis('equal')
        axs.plot(de[minind[0]],dn[minind[1]],'ro') # plot location
        axs.plot(locs[:,3],locs[:,4],'r^') # plot all the stations
        for kk in range(len(locs)): # plot all station names
            axs.text(locs[kk,3],locs[kk,4],str(int(locs[kk,0])))
            if (locs[kk,0] in staids):
                plt.plot(locs[kk,3],locs[kk,4],'g^')
        axs.set_title('Depth: '+str(dz[ii])+' m Amp: '+str(A0s[minind[3]]))
        axs.set_xlim((np.min(X),np.max(X)))
        axs.set_ylim((np.min(Y),np.max(Y)))
        fig.colorbar(cs)
    return None


def plot_fits(ampinputs,err,staamps,predamps,weights):
    fig,axs=plt.subplots(nrows=2,ncols=1,figsize=(10,8))
    axs[0].plot(ampinputs['station'].values,ampinputs['amp'].values,label='Amplitudes')
    axs[0].tick_params(labelrotation=90)
    fig.suptitle('%Err: '+str(err), fontsize=16)
    axs[1].plot(ampinputs['station'].values,staamps,label='AF corrected amplitudes')
    axs[1].plot(ampinputs['station'].values,predamps,label='Predicted amplitudes')
    axs[1].tick_params(labelrotation=90)
    ax2 = axs[1].twinx()
    ax2.plot(ampinputs['station'].values,weights,color=(0,0,0),label="weights")
    ax2.legend()
    axs[1].legend()
    return None


def get_coords():
    file='/Users/human/Dropbox/Research/Rattlesnake_Ridge/stingray_rr' \
         '/srOutput/0.6-0.75/srRays_38_2.mat'
    sta=sio.loadmat(file) # this doesn't work for v7.3 mat files

    # import h5py
    # import numpy as np
    # sta = {}
    # f = h5py.File(file)
    # for k, v in f.items():
    #     sta[k] = np.array(v)
    #
    # import hdf5storage
    # sta = hdf5storage.loadmat(file)

    # sta = mat73.loadmat(file)
    e0=sta['srRays']['srGeometry'][0][0][0][0][1][0][0]
    n0=sta['srRays']['srGeometry'][0][0][0][0][2][0][0]
    z0=sta['srRays']['srGeometry'][0][0][0][0][3][0][0]
    de=np.zeros(len(sta['srRays']['xg'][0][0]))
    dn=np.zeros(len(sta['srRays']['yg'][0][0]))
    dz=np.zeros(len(sta['srRays']['zg'][0][0]))
    # east range
    for ii in range(len(de)):
        de[ii]=sta['srRays']['xg'][0][0][ii][0]
    de+=e0
    # north range
    for ii in range(len(dn)):
        dn[ii]=sta['srRays']['yg'][0][0][ii][0]
    dn+=n0
    # vertical range
    for ii in range(len(dz)):
        dz[ii]=sta['srRays']['zg'][0][0][ii][0]
    dz+=z0
    elev=sta['srRays']['elevation'][0][0]
    de*=1000
    dn*=1000
    dz*=1000
    elev*=1000
    return de, dn, dz, elev


def event_stream(smallst,reft):
    start=reft
    finish=reft+0.2
    eventst=smallst.copy().slice(starttime=start,endtime=finish)
    return eventst


def process_data(st,reft):
    # datacentre = "IRIS"
    # client = Client(datacentre)
    start=reft-0.1
    finish=reft+0.4
    smallst=Stream()
    for tr in st:
        # print(tr.stats.station)
        smalltr=tr.copy().slice(starttime=start,endtime=finish)
        # # inv = Inventory()
        # tr.stats.location = "00"
        # # inv += client.get_stations(network=tr.stats.network,
        # #                            station=tr.stats.station,
        # #                            channel=tr.stats.channel,
        # #                            location=tr.stats.location,
        # #                            starttime=start,
        # #                            endtime=finish)
        # # response = inv.get_response(f"{tr.stats.network}."
        # #                             f"{tr.stats.station}.{tr.stats.location}"
        # #                             f".{tr.stats.channel}", start)
        # response = read_inventory(path_or_file_object=f"./resp/RESP."
        #                           f"{tr.stats.network}", format="RESP")
        # response.networks[0].code = tr.stats.network
        # response.networks[0].stations[0].code = tr.stats.station
        # response.networks[0].stations[0].channels[0].code = tr.stats.channel
        # response.networks[0].stations[0].channels[0].location_code = \
        #     tr.stats.location
        if tr.stats.network == 'UW':
            invfile = '/Users/human/Dropbox/Research/Rattlesnake_Ridge/amplitude_locations/resp/RESP.' + tr.stats.station + '.' + \
                      tr.stats.network + '..' + tr.stats.channel
        else:
            invfile = '/Users/human/Dropbox/Research/Rattlesnake_Ridge/amplitude_locations/resp/RESP.' + tr.stats.network + '.' + \
                      tr.stats.station + '..' + tr.stats.channel

        inv = read_inventory(path_or_file_object=invfile, format="RESP")

        try:
            smalltr.remove_response(inventory=inv, output='VEL')
        except:
            print('cannot correct for resp')
        else:
            smallst+=smalltr
    smallst=smallst.filter("bandpass", freqmin=20, freqmax=60, corners=4)
    smallst=smallst.detrend('linear')
    return smallst


def plot_traces(smallst, eventst, evd):
    fig,axs=plt.subplots(nrows=1,ncols=2,figsize=(8,10))
    labels=[]
    fac=5
    for jj in range(len(smallst)):
        #plt.plot(smallst[jj].times("relative"),smallst[jj].data/np.max(1.5*np.abs(smallst[jj].data))+jj,color=(0.5,0.5,0.5))
        axs[0].plot(smallst[jj].times("timestamp"),smallst[jj].data/np.max(fac*np.abs(smallst[0].data))+jj,color=(0.25,0.25,0.25))
        axs[0].plot(eventst[jj].times("timestamp"),eventst[jj].data/np.max(fac*np.abs(smallst[0].data))+jj,color=(96/256, 145/256, 186/256))
        axs[1].plot(smallst[jj].times("timestamp"),smallst[jj].data/np.max(fac*np.abs(smallst[jj].data))+jj,color=(0.25,0.25,0.25))
        axs[1].plot(eventst[jj].times("timestamp"),eventst[jj].data/np.max(fac*np.abs(smallst[jj].data))+jj,color=(96/256, 145/256, 186/256))
        if smallst[jj].stats.station in evd['station'].values:
            pt=UTCDateTime(evd[evd['station']==smallst[jj].stats.station].iloc[0]['picks']) #-smallst[0].times("utcdatetime")[0]
            axs[0].plot([pt, pt],[jj-0.4,jj+0.4], color=(0.5,0,0),alpha=0.75)
            axs[1].plot([pt, pt],[jj-0.4,jj+0.4], color=(0.5,0,0),alpha=0.75)
        labels.append(smallst[jj].stats.station+"-"+smallst[jj].stats.channel)
    axs[0].set_ylim((-1,41))
    axs[0].set_xlim((smallst[0].times("timestamp")[0],smallst[0].times("timestamp")[-1]))
    axs[0].set_xlabel('Time (s)')
    # axs.set_title("Event: "+str(ii)) #+" Ref. Station: "+str(refstaid))
    axs[0].set_yticks(range(len(labels)))
    axs[0].set_yticklabels(labels)
    axs[0].invert_yaxis()
    axs[1].set_ylim((-1,41))
    axs[1].set_xlim((smallst[0].times("timestamp")[0],smallst[0].times("timestamp")[-1]))
    axs[1].set_xlabel('Time (s)')
    # axs.set_title("Event: "+str(ii)) #+" Ref. Station: "+str(refstaid))
    axs[1].set_yticks(range(len(labels)))
    axs[1].set_yticklabels(labels)
    axs[1].invert_yaxis()
    return None


def dists_from_srrays():
    """ Calculates source-station distances from srRays stingray structures
    and saves to pickle files.

    TODO: write docstring

    Example:
        dists_from_srrays()

    """

    # load positions
    stas = pd.read_csv('amplitude_station_locations.csv', sep=',')
    stas = stas.sort_values(by=['latitude'], ascending=False)

    # load station coords
    # locs=np.loadtxt('ca_stalocs_1.dat')

    # # add distances to dataframe
    # hts=np.zeros(len(stas))
    # for ii in range(len(stas)):
    #     name=stas.iloc[ii]['station']
    #     if name=="UGAP3":
    #         name=103
    #     elif name=="UGAP5":
    #         name=105
    #     elif name=="UGAP6":
    #         name=106
    #     hts[ii]=locs[locs[:,0]==int(name)][0][5]

    stas['utmz'] = stas['elev']

    # location grid nodes
    de, dn, dz, elev = get_coords()

    # set up grid for plotting
    X, Y = np.meshgrid(de, dn)
    # plt.contourf(X,Y,elev,cmap='viridis')
    # plt.colorbar()

    # plt.figure()
    # plt.scatter(X,Y,s=20, marker='o', c=elev)

    beta = 400  # m/s
    f = 40  # frequency
    Q = 50  # quality factor (??)
    B = np.pi * f / (Q * beta)  # 1/m
    A0 = 1

    # Save distances
    for ii in range(len(stas)):
        print(
            f"Processing station {stas.iloc[ii]['station']}({ii + 1}/{len(stas)})")
        dists = np.zeros((len(de), len(dn), len(dz)))
        stax = stas.iloc[ii]['utmx']
        stay = stas.iloc[ii]['utmy']
        staz = stas.iloc[ii]['utmz']
        amps = np.zeros((len(de), len(dn), len(dz)))
        dists = np.zeros((len(de), len(dn), len(dz)))
        for jj, e in enumerate(de):
            for kk, n in enumerate(dn):
                z0 = elev[jj, kk]
                for ll, z in enumerate(dz):
                    r = np.sqrt((e - stax) ** 2 + (n - stay) ** 2 + (
                                (z0 + z) - staz) ** 2)
                    dists[jj, kk, ll] = r
                    # amps[kk,jj,ll]=A0*np.exp(-B*r)/r
        file = f'/Users/human/Dropbox/Research/Rattlesnake_Ridge' \
               f'/amplitude_locations/ray_tracing/dists_{stas.iloc[ii]["station"]}.pkl'
        with open(file, 'wb') as f:
            pickle.dump(dists, f)
        # fig,axs=plt.subplots(nrows=1,ncols=2,figsize=(10,6))
        # #plt.contourf(X,Y,dists[:,:,0],cmap='viridis')
        # im=axs[0].scatter(X,Y,s=20, marker='o', c=dists[:,:,0].T, vmin=0, vmax=400)
        # axs[0].plot(stax,stay,'r^')
        # axs[0].set_title("Station "+stas.iloc[ii]['station']+ " surface distances")
        # #plt.contourf(X,Y,dists[:,:,-1],cmap='viridis')
        # im=axs[1].scatter(X,Y,s=20, marker='o', c=dists[:,:,-1].T, vmin=0, vmax=400)
        # axs[1].plot(stax,stay,'r^')
        # axs[1].set_title("Station "+stas.iloc[ii]['station']+ " base distances")
        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # fig.colorbar(im, cax=cbar_ax)
        # plt.title("distance (m)")
        # fig.savefig('/Users/amt/Documents/rattlesnake_ridge/ray_tracing/dists_'+stas.iloc[ii]['station']+'.png')


def dijkstra_dists_from_srrays(plot=False):
    """ Calculates source-station Dijkstra distances from srRays iprec Stingray
    structure and saves distances to pickle files.

    # TODO:
    Returns:

    Example:

    """
    # FIXME: remove plot declaration after testing
    plot=False

    start = time.time()

    # load positions
    stas = pd.read_csv('amplitude_station_locations.csv', sep=',')
    stas = stas.sort_values(by=['latitude'], ascending=False)
    stas['utmz'] = stas['elev']

    # load node locations
    de, dn, dz, elev = get_coords()
    de = np.round(de)
    dn = np.round(dn)
    dz = np.round(dz)
    x, y = np.meshgrid(de, dn)

    # make array of node ids (note node0 is the station so this starts from node 1)
    node_ids = np.zeros((len(de), len(dn), len(dz)))
    c = 1
    for kk in range(len(dz)):
        for jj in range(len(dn)):
            for ii in range(len(de)):
                node_ids[ii, jj, kk] = c
                c += 1

    # load station file
    staid = 2
    # TODO: loop over all stations
    # for staid in stations:
    stax = stas[stas['station'] == str(staid)]['utmx'].values[0]
    stay = stas[stas['station'] == str(staid)]['utmy'].values[0]
    staz = stas[stas['station'] == str(staid)]['elev'].values[0]

    if isinstance(staid, str):
        stingray_id = "10" + str(staid[4])
        file = '/Users/human/Dropbox/Research/Rattlesnake_Ridge/stingray_rr' \
               '/srOutput/0.6-0.75/srRays_' + stingray_id + '_2.mat'
    else:
        file = '/Users/human/Dropbox/Research/Rattlesnake_Ridge/stingray_rr' \
               '/srOutput/0.6-0.75/srRays_' + str(staid) + '_2.mat'

    refsta = sio.loadmat(file)
    distances = np.empty((151, 251, 76))
    iprec = refsta['srRays']['iprec'][0][0]

    # plot trace paths
    if plot:
        fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(6,12))
        tmp=ax.pcolor(x,y,node_ids[:,:,0].T,cmap='viridis',vmin=0,vmax=37901)
        ax.plot(stax,stay,'k^',markersize=10)
        ax.set_aspect('equal')
        plt.show()

    dds = np.zeros_like(iprec).astype('float')
    for current_node in tqdm(range(1, iprec.size + 1)):
        ii0 = np.where(node_ids == current_node)[0][0]
        jj0 = np.where(node_ids == current_node)[1][0]
        kk0 = np.where(node_ids == current_node)[2][0]
        xc = []
        yc = []
        zc = []
        iss = []
        jss = []
        kss = []
        while current_node != 0:
            # print(current_node)
            # get current node index location and append
            ii = np.where(node_ids == current_node)[0][0]
            jj = np.where(node_ids == current_node)[1][0]
            kk = np.where(node_ids == current_node)[2][0]
            iss.append(ii)
            jss.append(jj)
            kss.append(kk)
            # get current node physical location
            xc.append(de[ii])
            yc.append(dn[jj])
            zc.append(dz[kk] + elev[ii, jj])
            if dds[ii, jj, kk] != 0:
                # print('distance for node ' + str(
                #     current_node) + ' is already calculated')
                break
            # get previous node
            prev_node = iprec[ii, jj, kk]
            # make previous location current location
            current_node = prev_node
            # # trying to speed things up by checking we havent already counted the distance for this node
        if current_node == 0:
            # append station location
            xc.append(stax)
            yc.append(stay)
            zc.append(staz)
            # convert to arrays
        xc = np.asarray(xc)
        yc = np.asarray(yc)
        zc = np.asarray(zc)
        # calc distances between nodes
        da = xc[1:] - xc[:-1]
        db = yc[1:] - yc[:-1]
        dc = zc[1:] - zc[:-1]
        dds_along_path = np.cumsum(np.sqrt(da ** 2 + db ** 2 + dc ** 2)[::-1])[
                         ::-1]
        if dds[ii, jj, kk] != 0:
            dds_along_path += dds[ii, jj, kk]
        for a, b, c, d in zip(iss, jss, kss, dds_along_path):
            dds[a, b, c] = d

    # save results to a pickle file
    file = f'/Users/human/Dropbox/Research/Rattlesnake_Ridge' \
           f'/amplitude_locations/ray_tracing/dists_{staid}.pkl'
    with open(file, 'wb') as f:
        pickle.dump(dds, f)

    if plot:
        xc=np.array(xc)
        yc=np.array(yc)
        zc=np.array(zc)
        cbar=plt.scatter(xc,yc,s=20,c=dds,vmin=0,vmax=750,cmap='jet',zorder=10)
        plt.colorbar(cbar)
        plt.show()

    # display the runtime
    end = time.time()
    hours = int((end - start) / 60 / 60)
    minutes = int(((end - start) / 60) - (hours * 60))
    seconds = int((end - start) - (minutes * 60) - (hours * 60 * 60))
    print(f"Runtime: {hours} h {minutes} m {seconds} s")

    return None

def srrays_to_pickle():
    """ Converts srRays Stingray structures to pickle files.

    # TODO: write docstring

    Example:
        srrays_to_pickle()

    """

    # load positions
    stas = pd.read_csv('amplitude_station_locations.csv', sep=',')
    stas = stas.sort_values(by=['latitude'], ascending=False)
    # stas=stas.reset_index(drop=True)
    # stas=stas.drop([22])
    # stas=stas.drop([29])

    for refstaid in stas['station']:
        print(refstaid)
        if refstaid[:4] == "UGAP":
            stingray_id = "10" + str(refstaid[4])
            file = '/Users/human/Dropbox/Research/Rattlesnake_Ridge/stingray_rr' \
                   '/srOutput/OSOS_0.5/srRays_' + stingray_id + '_2.mat'
        else:
            file = '/Users/human/Dropbox/Research/Rattlesnake_Ridge/stingray_rr' \
                   '/srOutput/OSOS_0.5/srRays_' + str(refstaid) + '_2.mat'
        refsta = sio.loadmat(file)
        # reftts=np.empty((151,251,76))
        reftts = refsta['srRays']['time'][0][0]

        # # why is the depth grid truncated?
        # reftts=reftts[:,:,:40]

        with open('/Users/human/Dropbox/Research/Rattlesnake_Ridge'
                  '/amplitude_locations/ray_tracing/tts_' + str(
            refstaid) + '.pkl', 'wb') as f:
            pickle.dump(reftts, f)

        # # get dijkstra distances
        # refprec = refsta['srRays']['iprec'][0][0]
        #
        # with open('/Users/human/Dropbox/Research/Rattlesnake_Ridge'
        #           '/amplitude_locations/ray_tracing/dijkstra_dists_' + str(
        #     refstaid) + '.pkl', 'wb') as f:
        #     pickle.dump(refprec, f)


def amplitude_locations():
    """ Driver function to calculate amplitude-based locations.

    # TODO: finish docstring

    Example:
        amplitude_locations()

    """
    # load positions
    stas = pd.read_csv('amplitude_station_locations.csv', sep=',')
    stas = stas.sort_values(by=['latitude'], ascending=False)

    # load amplitude factors and add to stas
    afs = pd.read_csv('noise_metrics.csv')
    meanaf = np.zeros(len(stas))
    medaf = np.zeros(len(stas))
    rmsaf = np.zeros(len(stas))
    for ii in range(len(stas)):
        meanaf[ii] = afs[afs['station'] == stas.iloc[ii]['station']][
            'mean'].values
        medaf[ii] = afs[afs['station'] == stas.iloc[ii]['station']][
            'median'].values
        rmsaf[ii] = afs[afs['station'] == stas.iloc[ii]['station']][
            'rms'].values
    stas['meanaf'] = meanaf
    stas['medaf'] = medaf
    stas['rmsaf'] = rmsaf

    # # load the data
    date = UTCDateTime(2018, 3, 13)
    st = Stream()
    print('loading data')
    for sta in stas['station'].values:
        #

        file = '/Users/human/Desktop/RR_MSEED/5A.' + str(
            sta) + '..DP1.2018-' + str(date.month).zfill(2) + '-' + str(
            date.day).zfill(2) + 'T00.00.00.ms'
        print(file)
        try:
            st += read(file)
        except:
            try:
                file = '/Users/human/Desktop/RR_MSEED/5A.' + str(
                    sta) + '..EHN.2018-' + str(date.month).zfill(
                    2) + '-' + str(
                    date.day).zfill(2) + 'T00.00.00.ms'
                st += read(file)
            except:
                print('No data for ' + file)

    # add lat lon data
    for tr in st:
        # get the station
        tr_id = tr.stats.station

        # set new id in stream
        # tr.stats.station = tr_id
        tr.stats.coordinates = AttribDict(
            {'latitude': stas[stas.station == tr_id].latitude.values[0],
             'longitude': stas[stas.station == tr_id].longitude.values[0]})

    # location grid nodes
    de, dn, dz, elev = get_coords()
    # fac=1
    # de=de[::fac]
    # dn=dn[::fac]
    # dz=dz[:41:fac]
    # elev=elev[::fac,::fac]

    # set up grid for plotting
    X, Y = np.meshgrid(de, dn)

    # load tylers phase picks -- phase.mrkr
    df = pd.read_csv("res.mrkr", skiprows=1, delim_whitespace=True,
                     usecols=[1, 2, 4, 5, 6, 7, 8],
                     names=['pick date', 'pick time', 'station_channel',
                            'hash', 'marker date', 'marker time', 'phase'])

    # grab only P picks
    df = df[df['phase'] == 'P']

    # define the stations and channels
    df['station'] = df['station_channel'].str.split('.').str[1]
    df['channel'] = df['station_channel'].str.split('.').str[-1]

    # convert markers and picks to date format
    df['markers'] = pd.to_datetime(df['marker date'] + ' ' + df['marker time'],
                                   errors='coerce', format='%Y-%m-%d %H:%M:%S')
    df['picks'] = pd.to_datetime(df['pick date'] + ' ' + df['pick time'],
                                 errors='coerce', format='%Y-%m-%d %H:%M:%S')
    markers = pd.unique(df['markers'])
    hashes = pd.unique(df['hash'])

    evlocs = []
    # # for each marker
    # markers=markers[:2]
    for ii, markertime in enumerate(markers):
        print(f'Processing marker {ii + 1} of {len(markers)}')

        # find all the picks that correspond to that marker
        evd = df[df['markers'] == markertime]

        # calculate the maximum differential time
        dt = (np.max(evd['picks']) - np.min(evd['picks'])).total_seconds()

        # crude QC to make sure there are more than 5 picks and the time between them is reasonable
        if len(evd) > 5 and dt < 0.5:
            staids = evd['station'].values
            stadts = (evd['picks'] - np.min(
                evd['picks'])).dt.total_seconds().values

            # cut, correct, filter, and detrend traces
            stalist = stas['station'].values.tolist()
            reft = UTCDateTime(np.min(evd['picks']))
            smallst = process_data(st, reft)

            # event only stream
            eventst = event_stream(smallst, reft)

            # prep inputs for amplocation
            ampinputs = pd.DataFrame(columns=['station', 'ampfac', 'amp'])
            for jj in range(len(smallst)):
                # # if stations are in pick list, add amplitudes
                # if smallst[jj].stats.station in evd['station'].values:
                #     ampinputs.loc[jj] = [eventst[jj].stats.station, stas[stas['station']==eventst[jj].stats.station]['meanaf'].values[0]/stas.iloc[0]['meanaf'], np.sqrt(np.mean((eventst[jj].data)**2))]
                rms = np.sqrt(np.mean((eventst[jj].data) ** 2))
                ampinputs.loc[jj] = [eventst[jj].stats.station, stas[
                    stas['station'] == eventst[jj].stats.station][
                    'rmsaf'].values[0] / stas.iloc[0]['rmsaf'], rms]

            # plt.figure()
            normamps = ampinputs['amp'].values / ampinputs['ampfac'].values
            ind = np.where(normamps == np.max(normamps))[0][0]
            maxstax = stas[ampinputs.iloc[ind]['station'] == stas['station']][
                'utmx'].values[0]
            maxstay = stas[ampinputs.iloc[ind]['station'] == stas['station']][
                'utmy'].values[0]
            dfrommax = np.zeros(len(ampinputs))
            for kk in range(len(ampinputs)):
                stax = stas[ampinputs.iloc[kk]['station'] == stas['station']][
                    'utmx'].values[0]
                stay = stas[ampinputs.iloc[kk]['station'] == stas['station']][
                    'utmy'].values[0]
                dfrommax[kk] = np.sqrt(
                    (maxstax - stax) ** 2 + (maxstay - stay) ** 2)
            ampinputs['dfrommax'] = dfrommax

            # # locate me with travel times
            # refstaid=staids[0]
            # resid=locate(refstaid,staids,stadts)
            # minind = np.unravel_index(np.argmin(resid, axis=None), resid.shape)

            # locate me with amplitudes
            A0s = np.logspace(-5, -3, 11)
            ampresid, staamps, predamps, err, weights \
                = amplocate(ampinputs['station'].values,
                                           ampinputs['ampfac'].values,
                                           ampinputs['amp'].values, de, dn, dz,
                                           A0s, ampinputs['dfrommax'].values,
                                           2, dijkstra=1)
            minind = np.unravel_index(np.argmin(ampresid, axis=None),
                                      ampresid.shape)

            mininds = np.where(ampresid <= ampresid[
                minind[0], minind[1], minind[2], minind[3]] * 1.05)

            plots = 0
            if plots:
                # plot residual
                plot_resid(X, Y, ampresid, minind, mininds, de,
                                          dn, dz, stas, staids, A0s)

                # plot fits
                plot_fits(ampinputs, err, staamps, predamps,
                                         weights)

            evlocs.append([hashes[ii], de[minind[0]], dn[minind[1]],
                           dz[minind[2]], A0s[minind[3]], err, len(evd)])

    # plot locs
    evlocs = pd.DataFrame(evlocs, columns=['ID', 'x', 'y', 'z', 'amp', 'error',
                                           'picks'])
    plt.figure()
    plt.scatter(evlocs['x'].values, evlocs['y'].values, s=15,
                c=evlocs['error'].values)
    plt.plot(stas['utmx'].values, stas['utmy'].values,
             'r^')  # plot all the stations
    for ii in range(len(stas)):
        plt.text(stas.iloc[kk]['utmx'], stas.iloc[kk]['utmy'],
                 stas.iloc[kk]['station'], color='r')
    plt.colorbar()
    plt.axis('equal')
    plt.show()

    # plot depth slices
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=((12, 8)))
    low = evlocs[(evlocs['y'] > 5155500) & (evlocs['y'] <= 5155600)]
    im = ax[0].scatter(low['x'].values, low['z'].values, s=15,
                       c=low['error'].values, vmin=0, vmax=100)
    low = evlocs[(evlocs['y'] > 5155600) & (evlocs['y'] <= 5155700)]
    im = ax[1].scatter(low['x'].values, low['z'].values, s=15,
                       c=low['error'].values, vmin=0, vmax=100)
    low = evlocs[(evlocs['y'] > 5155700) & (evlocs['y'] <= 5155800)]
    im = ax[2].scatter(low['x'].values, low['z'].values, s=15,
                       c=low['error'].values, vmin=0, vmax=100)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    ax[1].set_ylim((-81, 1))
    ax[2].set_ylim((-81, 1))
    plt.show()

    # plot low error locs
    evlocs = pd.DataFrame(evlocs, columns=['ID', 'x', 'y', 'z', 'amp', 'error',
                                           'picks'])
    plt.figure()
    low = evlocs[(evlocs['error'] <= 30)]
    plt.scatter(low['x'].values, low['y'].values, s=15, c=low['error'].values)
    plt.plot(stas['utmx'].values, stas['utmy'].values,
             'r^')  # plot all the stations
    for ii in range(len(stas)):
        plt.text(stas.iloc[kk]['utmx'], stas.iloc[kk]['utmy'],
                 stas.iloc[kk]['station'], color='r')
    plt.colorbar()
    plt.axis('equal')
    plt.show()

    # plot low error depth slices
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=((12, 8)))
    low = evlocs[(evlocs['y'] > 5155500) & (evlocs['y'] <= 5155600) & (
            evlocs['error'] <= 30)]
    im = ax[0].scatter(low['x'].values, low['z'].values, s=15,
                       c=low['error'].values, vmin=0, vmax=100)
    ax[0].set_ylim((-81, 1))
    low = evlocs[(evlocs['y'] > 5155600) & (evlocs['y'] <= 5155700) & (
            evlocs['error'] <= 30)]
    im = ax[1].scatter(low['x'].values, low['z'].values, s=15,
                       c=low['error'].values, vmin=0, vmax=100)
    ax[1].set_ylim((-81, 1))
    low = evlocs[(evlocs['y'] > 5155700) & (evlocs['y'] <= 5155800) & (
            evlocs['error'] <= 30)]
    im = ax[2].scatter(low['x'].values, low['z'].values, s=15,
                       c=low['error'].values, vmin=0, vmax=100)
    ax[2].set_ylim((-81, 1))
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()

    evlocs.to_csv('res.locs', index=False)

    # export files to plot in GMT
    hypocenters = []
    hypos = evlocs[['x', 'y', 'z', 'error']]
    for index, entry in hypos.iterrows():
        hypocenters.append([entry['x'], entry['y'], entry['z'],
                            entry['error']])

    # generate files for plotting hypocenters via gmt
    with open(f"x_y_horizUncert_amplitudeLocs.csv", "w") as write_file:
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
            lat, lon = utm.to_latlon(hypocenter[0], hypocenter[1], 10, 'N')

            # scale horizontal uncertainty for plotting
            transformed_uncerty = ((hypocenter[3] - uncerty_min) * new_range \
                                   / uncerty_range) + new_min
            line = f"{lon} {lat} {transformed_uncerty}\n"
            write_file.write(line)

    with open(f"x_z_amplitudeLocs.csv", "w") as write_file:
        # write header
        write_file.write("LON Z\n")

        # write each hypocenter to file
        for hypocenter in hypocenters:
            lat, lon = utm.to_latlon(hypocenter[0], hypocenter[1], 10, 'N')

            line = f"{lon} {hypocenter[2]}\n"
            write_file.write(line)