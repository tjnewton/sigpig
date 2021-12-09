"""
Functions to generate various figures. 
"""

import obspy
from obspy import read, Stream, Trace
from obspy.core.utcdatetime import UTCDateTime
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.dates import DateFormatter, AutoDateLocator
import glob
import numpy as np
import scipy.signal as spsig
from data import max_amplitude, snr
import pickle
import pandas as pd

# FIXME ::: Add elevation functions from Desktop/autopicker/preprocessing/station_locations.py
# above requires configuring environment with rasterio and dependencies


def nearest(items: np.ndarray, pivot):
    return min(items, key=lambda x: abs(x - pivot))


def nearest_ind(items: np.ndarray, pivot):
    time_diff = np.abs([date - pivot for date in items])
    return time_diff.argmin(0)


def plot_stalta(obspy_Trace: obspy.core.trace.Trace, cft: np.ndarray,
                start_Time: UTCDateTime, end_Time: UTCDateTime) -> None:
    """
    Plots a timeseries from an Obspy trace and the corresponding STA/LTA for the specified time period.

    Parameters
    ----------
    obspy_Trace: Obspy trace object
    cft: np.ndarray of characteristic function returned by obspy.signal.trigger.classic_sta_lta
    start_Time: UTCDateTime object
    end_Time: UTCDateTime object

    Example
    -------
    start_Time = UTCDateTime("2018-03-13T01:32:13.5Z")
    end_Time = UTCDateTime("2018-03-13T01:32:15.5Z")
    plot_stalta(trace, cft, start_Time, end_Time)

    """
    # set the figure size
    figureWidth = 19
    figureHeight = 5

    # find start and end index for subset within the time bounds of the trace
    subset_start_idx = nearest_ind(np.array(trace.times(type="utcdatetime")),
                                   start_Time)
    subset_end_idx = nearest_ind(np.array(trace.times(type="utcdatetime")),
                                 end_Time)
    subset_times = np.array(trace.times(type="matplotlib"))[
                   subset_start_idx:subset_end_idx + 1]
    subset_data = trace.data[subset_start_idx:subset_end_idx + 1]

    # plot STA/LTA over time & compare with time series
    maxTraceValue, _ = max_amplitude(subset_data)
    fig, ax = plt.subplots(nrows=2, sharex=True,
                           figsize=(figureWidth, figureHeight))
    ax[0].plot_date(subset_times, subset_data, fmt="k-", linewidth=0.4)
    ax[0].set_ylabel("Amplitude")
    ax[1].plot_date(subset_times, cft[subset_start_idx:subset_end_idx + 1],
                    fmt="k-", linewidth=0.4)
    ax[1].set_ylabel(f"STA:{sta} / LTA:{lta}")
    plt.xlabel("Time")
    plt.show()

    return None


def plot_Time_Series_And_Spectrogram(doi, doi_end, files_path, filter=False,
                                     bandpass=[], time_markers={}):
    """
    Plots time series and spectrograms for all files in the `files_path` for
    the specified time period starting at `doi` and ending at `doi_end`. Data
    are first bandpass filtered if `filter=True`.

    Example:
        # define dates of interest
        # doi = UTCDateTime("2016-09-26T07:30:00.0Z") # period start
        # doi_end = UTCDateTime("2016-09-26T10:30:00.0Z") # period end
        # doi = UTCDateTime("2016-09-26T08:52:00.0Z") # period start
        # doi_end = UTCDateTime("2016-09-26T08:55:00.0Z") # period end
        # doi = UTCDateTime("2016-09-26T09:25:00.0Z") # period start
        # doi_end = UTCDateTime("2016-09-26T09:27:00.0Z") # period end
        # doi = UTCDateTime("2016-09-26T09:28:00.0Z") # period start
        # doi_end = UTCDateTime("2016-09-26T09:30:00.0Z") # period end
        doi = UTCDateTime("2016-09-26T09:23:00.0Z") # period start
        doi_end = UTCDateTime("2016-09-26T09:33:00.0Z") # period end

        # define time series files path
        files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/picked"

        # bandpass filter from 2-8 Hz
        filter = True
        bandpass = [1, 15]

        # plot with time markers on specified stations
        time_markers = {"AK.GLB": [UTCDateTime("2016-09-26T09:25:50.0Z"),
                                    UTCDateTime("2016-09-26T09:26:05.0Z")],
                        "AK.PTPK": [UTCDateTime("2016-09-26T09:26:05.0Z"),
                                    UTCDateTime("2016-09-26T09:26:20.0Z")],
                        "AV.WASW": [UTCDateTime("2016-09-26T09:25:46.0Z"),
                                    UTCDateTime("2016-09-26T09:26:01.0Z")],
                        "YG.MCR4": [UTCDateTime("2016-09-26T09:25:53.0Z"),
                                    UTCDateTime("2016-09-26T09:26:08.0Z")],
                        "YG.NEB3": [UTCDateTime("2016-09-26T09:25:54.5Z"),
                                    UTCDateTime("2016-09-26T09:26:09.5Z")],
                        "YG.MCR1": [UTCDateTime("2016-09-26T09:25:54.5Z"),
                                    UTCDateTime("2016-09-26T09:26:09.5Z")],
                        "YG.RH08": [UTCDateTime("2016-09-26T09:26:02.5Z"),
                                    UTCDateTime("2016-09-26T09:26:17.5Z")],
                        "YG.RH10": [UTCDateTime("2016-09-26T09:26:01.5Z"),
                                    UTCDateTime("2016-09-26T09:26:16.5Z")],
                        "YG.RH09": [UTCDateTime("2016-09-26T09:26:01.5Z"),
                                    UTCDateTime("2016-09-26T09:26:16.5Z")],
                        "AV.WACK": [UTCDateTime("2016-09-26T09:25:49.5Z"),
                                    UTCDateTime("2016-09-26T09:26:04.5Z")],
                        "YG.NEB1": [UTCDateTime("2016-09-26T09:25:56.5Z"),
                                    UTCDateTime("2016-09-26T09:26:11.5Z")],
                        "TA.N25K": [UTCDateTime("2016-09-26T09:25:49.5Z"),
                                    UTCDateTime("2016-09-26T09:26:04.5Z")],
                        "YG.MCR3": [UTCDateTime("2016-09-26T09:25:49.5Z"),
                                    UTCDateTime("2016-09-26T09:26:04.5Z")],
                        "AK.KLU": [UTCDateTime("2016-09-26T09:26:07.0Z"),
                                    UTCDateTime("2016-09-26T09:26:22.0Z")],
                        "YG.MCR2": [UTCDateTime("2016-09-26T09:25:47.5Z"),
                                    UTCDateTime("2016-09-26T09:26:02.5Z")],
                                    }

        fig = plot_Time_Series_And_Spectrogram(doi, doi_end, files_path,
                                               filter=filter,
                                               bandpass=bandpass,
                                               time_markers=time_markers)

    Another example:
        # define time period
        doi = UTCDateTime("2016-09-26T09:23:00.0Z") # period start
        doi_end = UTCDateTime("2016-09-26T09:33:00.0Z") # period end

        # define time series files path
        files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/picked"

        # bandpass filter from 2-8 Hz
        filter = True
        bandpass = [1, 15]

        # build time markers from snuffler marker file
        marker_file_path = "lfe_template.mrkr"
        prepick_offset = 11 # in seconds
        templates, station_dict, time_markers = markers_to_template(
                                                              marker_file_path,
                                                              prepick_offset,
                                                             time_markers=True)

        fig = plot_Time_Series_And_Spectrogram(doi, doi_end, files_path,
                                               filter=filter,
                                               bandpass=bandpass,
                                               time_markers=time_markers)

    """

    # find all files for specified day
    day_file_list = sorted(glob.glob(f"{files_path}/*N.{doi.year}"
                                     f"-{doi.month:02}-{doi.day:02}.ms"))
    # load files into stream
    st = Stream()
    for file in day_file_list:
        st += read(file)

        # filter before trimming to avoid edge effects
        if filter:
            # bandpass filter specified frequencies
            st.filter('bandpass', freqmin=bandpass[0], freqmax=bandpass[1])

        # take the specified time period + and - 30 seconds for edge effects
        st.trim(doi - 30, doi_end + 30)

    # initialize figure and set the figure size
    figureWidth = 50
    figureHeight = 3.5 * len(st)  # 0.6 for all stations # 3.5
    fig = plt.figure(figsize=(figureWidth, figureHeight))
    gs = fig.add_gridspec(3, 1)
    amplitude_plot = fig.add_subplot(gs[0, :])
    frequency_plot = fig.add_subplot(gs[1:, :])
    # make a subplot of subplots for spectrograms
    frequency_plots = gridspec.GridSpecFromSubplotSpec(len(st), 1,
                                                       subplot_spec=frequency_plot)
    # amplitude_plot = plt.subplot(2, 1, 1)
    # frequency_plot = plt.subplot(2, 1, 2)

    # loop through stream and generate plots
    y_labels = []
    for index, trace in enumerate(st):
        # find max trace value for normalization
        maxTraceValue, _ = max_amplitude(trace)

        # define data to work with
        time = trace.times("matplotlib")
        trace_start = trace.stats.starttime
        norm_amplitude = (trace.data - np.min(trace.data)) / (maxTraceValue -
                                                              np.min(
                                                                  trace.data)) * 1.25 + index
        # add trace to waveform plot
        amplitude_plot.plot_date(time, norm_amplitude, fmt="k-", linewidth=0.7)

        # plot time markers for this trace if they exist
        network_station = f"{trace.stats.network}.{trace.stats.station}"
        if network_station in time_markers:
            time_marker = time_markers[network_station]
            # plot time_marker box
            x_vals = [time_marker[0].matplotlib_date,
                      time_marker[0].matplotlib_date,
                      time_marker[1].matplotlib_date,
                      time_marker[1].matplotlib_date,
                      time_marker[0].matplotlib_date]
            y_vals = [min(norm_amplitude), max(norm_amplitude),
                      max(norm_amplitude), min(norm_amplitude),
                      min(norm_amplitude)]
            amplitude_plot.plot_date(x_vals, y_vals,
                                     fmt="r-", linewidth=2.0)

        # add station name to list of y labels
        y_labels.append(f"{network_station}.{trace.stats.channel}")

        # print(trace.stats.sampling_rate)

        # build information for spectrogram
        duration = (doi_end + 30) - (doi - 30)
        num_windows = (duration / 60) * 40 - 2
        window_duration = duration / num_windows
        window_length = int(window_duration * trace.stats.sampling_rate)
        nfftSTFT = window_length * 2  # nyquist
        overlap = int(
            window_length / 2)  # overlap between spec. windows in samples
        [fSTFT, tSTFT, STFT] = spsig.spectrogram(trace.data,
                                                 trace.stats.sampling_rate,
                                                 nperseg=window_length,
                                                 noverlap=overlap,
                                                 nfft=nfftSTFT)
        # plot the spectrogram
        spec = fig.add_subplot(frequency_plots[(index + 1) * -1, 0])

        # dB = 20*log() convention
        spec.pcolormesh(tSTFT, fSTFT, 20 * np.log10(np.absolute(STFT)),
                        cmap='magma')

        # plot time markers for this trace if they exist
        if network_station in time_markers:
            # plot time_marker box, transform x_vals to s for spectrogram
            x_vals = [time_marker[0] - trace_start,
                      time_marker[0] - trace_start,
                      time_marker[1] - trace_start,
                      time_marker[1] - trace_start,
                      time_marker[0] - trace_start]
            # transform y values for spectrogram
            y_vals = [fSTFT.min(), fSTFT.max(),
                      fSTFT.max(), fSTFT.min(),
                      fSTFT.min()]
            spec.plot(x_vals, y_vals, "r-", linewidth=2.0)

        spec.set_xlim([30, duration - 30])
        spec.set_ylabel(f"{trace.stats.network}.{trace.stats.station}."
                        f"{trace.stats.channel}",
                        rotation=0, labelpad=40)
        spec.tick_params(axis='x', which='both', bottom=False, top=False,
                         labelbottom=False)
    # spec.set_yticks([])

    # set axes attributes
    amplitude_plot.set_yticks(np.arange(0.5, len(st) + 0.5))
    amplitude_plot.set_yticklabels(y_labels)
    amplitude_plot.set_ylabel('Station.Channel')
    amplitude_plot.set_xlim([doi.matplotlib_date, doi_end.matplotlib_date])
    amplitude_plot.set_xlabel(f'Time: Hr:Min:Sec of {doi.month:02}-'
                              f'{doi.day:02}-{doi.year}')
    myFmt = DateFormatter("%H:%M:%S")  # "%H:%M:%S.f"
    amplitude_plot.xaxis.set_major_formatter(myFmt)
    locator_x = AutoDateLocator(minticks=10, maxticks=35)
    amplitude_plot.xaxis.set_major_locator(locator_x)
    amplitude_plot.set_ylim((0, len(st) + 0.5))
    # frequency_plot.set_ylabel('Frequency (Hz)')
    # frequency_plot.set_xlabel('Time (s)')
    frequency_plot.set_yticks([])
    frequency_plot.set_xticks([])
    # fig.tight_layout()
    fig.savefig(f"{doi.month:02}-{doi.day:02}-{doi.year}T{doi.hour:02}."
                f"{doi.minute:02}.png", dpi=100)

    plt.show()


# plot time series for specified dates of interest from specified files
def plot_Time_Series(doi, doi_end, files_path, filter=False, bandpass=[],
                     time_markers={}):
    """
    Plots time series for all files in the `files_path` for
    the specified time period starting at `doi` and ending at `doi_end`. Data
    are first bandpass filtered if `filter=True`.

    Example:
        # define dates of interest
        # doi = UTCDateTime("2016-09-26T09:25:30.0Z") # period start
        # doi_end = UTCDateTime("2016-09-26T09:26:30.0Z") # period end
        doi = UTCDateTime("2016-09-26T09:27:30.0Z") # period start
        doi_end = UTCDateTime("2016-09-26T09:30:30.0Z") # period end

        # doi = UTCDateTime("2016-09-26T09:08:00.0Z") # period start
        # doi_end = UTCDateTime("2016-09-26T09:15:00.0Z") # period end
        # doi = UTCDateTime("2016-09-26T08:51:00.0Z") # period start
        # doi_end = UTCDateTime("2016-09-26T09:30:00.0Z") # period end

        # define time series files path
        # files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/picked"
        files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/2016-09-26"

        # bandpass filter from 1-15 Hz
        filter = True
        bandpass = [1, 15]

        # plot time markers on specified stations
        # time_markers = {"AK.GLB": [UTCDateTime("2016-09-26T09:25:50.0Z"),
        # 					 		UTCDateTime("2016-09-26T09:26:05.0Z")],
        # 				"AK.PTPK": [UTCDateTime("2016-09-26T09:26:05.0Z"),
        # 							UTCDateTime("2016-09-26T09:26:20.0Z")],
        # 				"AV.WASW": [UTCDateTime("2016-09-26T09:25:46.0Z"),
        # 					 		UTCDateTime("2016-09-26T09:26:01.0Z")],
        # 				"YG.MCR4": [UTCDateTime("2016-09-26T09:25:53.0Z"),
        # 					 		UTCDateTime("2016-09-26T09:26:08.0Z")],
        # 				"YG.NEB3": [UTCDateTime("2016-09-26T09:25:54.5Z"),
        # 					 		UTCDateTime("2016-09-26T09:26:09.5Z")],
        # 				"YG.MCR1": [UTCDateTime("2016-09-26T09:25:54.5Z"),
        # 					 		UTCDateTime("2016-09-26T09:26:09.5Z")],
        # 				"YG.RH08": [UTCDateTime("2016-09-26T09:26:02.5Z"),
        # 					 		UTCDateTime("2016-09-26T09:26:17.5Z")],
        # 				"YG.RH10": [UTCDateTime("2016-09-26T09:26:01.5Z"),
        # 					 		UTCDateTime("2016-09-26T09:26:16.5Z")],
        # 				"YG.RH09": [UTCDateTime("2016-09-26T09:26:01.5Z"),
        # 					 		UTCDateTime("2016-09-26T09:26:16.5Z")],
        # 				"AV.WACK": [UTCDateTime("2016-09-26T09:25:49.5Z"),
        # 					 		UTCDateTime("2016-09-26T09:26:04.5Z")],
        # 				"YG.NEB1": [UTCDateTime("2016-09-26T09:25:56.5Z"),
        # 					 		UTCDateTime("2016-09-26T09:26:11.5Z")],
        # 				"TA.N25K": [UTCDateTime("2016-09-26T09:25:49.5Z"),
        # 					 		UTCDateTime("2016-09-26T09:26:04.5Z")],
        # 				"YG.MCR3": [UTCDateTime("2016-09-26T09:25:49.5Z"),
        # 					 		UTCDateTime("2016-09-26T09:26:04.5Z")],
        # 				"AK.KLU": [UTCDateTime("2016-09-26T09:26:07.0Z"),
        # 					 		UTCDateTime("2016-09-26T09:26:22.0Z")],
        # 				"YG.MCR2": [UTCDateTime("2016-09-26T09:25:47.5Z"),
        # 					 		UTCDateTime("2016-09-26T09:26:02.5Z")],
        # 					 		}
        time_markers = {"YG.NEB3": [UTCDateTime("2016-09-26T09:28:46.5Z"),
                                    UTCDateTime("2016-09-26T09:29:00.5Z")],
                        "YG.NEB2": [UTCDateTime("2016-09-26T09:28:46.5Z"),
                                    UTCDateTime("2016-09-26T09:29:00.5Z")],
                        "YG.NEB1": [UTCDateTime("2016-09-26T09:28:47.5Z"),
                                    UTCDateTime("2016-09-26T09:29:00.5Z")],
                        "AK.GLB": [UTCDateTime("2016-09-26T09:28:40.0Z"),
                                    UTCDateTime("2016-09-26T09:28:54.0Z")],
                        "AK.PTPK": [UTCDateTime("2016-09-26T09:28:55.0Z"),
                                    UTCDateTime("2016-09-26T09:29:09.0Z")],
                        "AV.WASW": [UTCDateTime("2016-09-26T09:28:36.0Z"),
                                    UTCDateTime("2016-09-26T09:28:50.0Z")],
                        "YG.MCR4": [UTCDateTime("2016-09-26T09:28:47.0Z"),
                                    UTCDateTime("2016-09-26T09:29:01.0Z")],
                        "YG.MCR1": [UTCDateTime("2016-09-26T09:25:54.5Z"),
                                    UTCDateTime("2016-09-26T09:26:09.5Z")],
                        "YG.RH08": [UTCDateTime("2016-09-26T09:26:02.5Z"),
                                    UTCDateTime("2016-09-26T09:26:17.5Z")],
                        "YG.RH10": [UTCDateTime("2016-09-26T09:26:01.5Z"),
                                    UTCDateTime("2016-09-26T09:26:16.5Z")],
                        "YG.RH09": [UTCDateTime("2016-09-26T09:26:01.5Z"),
                                    UTCDateTime("2016-09-26T09:26:16.5Z")],
                        "AV.WACK": [UTCDateTime("2016-09-26T09:25:49.5Z"),
                                    UTCDateTime("2016-09-26T09:26:04.5Z")],
                        "YG.NEB1": [UTCDateTime("2016-09-26T09:25:56.5Z"),
                                    UTCDateTime("2016-09-26T09:26:11.5Z")],
                        "TA.N25K": [UTCDateTime("2016-09-26T09:25:49.5Z"),
                                    UTCDateTime("2016-09-26T09:26:04.5Z")],
                        "YG.MCR3": [UTCDateTime("2016-09-26T09:25:49.5Z"),
                                    UTCDateTime("2016-09-26T09:26:04.5Z")],
                        "AK.KLU": [UTCDateTime("2016-09-26T09:26:07.0Z"),
                                    UTCDateTime("2016-09-26T09:26:22.0Z")],
                        "YG.MCR2": [UTCDateTime("2016-09-26T09:25:47.5Z"),
                                    UTCDateTime("2016-09-26T09:26:02.5Z")],
                                    }

        fig = plot_Time_Series(doi, doi_end, files_path, filter=filter,
                               bandpass=bandpass, time_markers=time_markers)

        # plot different date range without time markers
        doi = UTCDateTime("2016-09-26T09:28:00.0Z") # period start
        doi_end = UTCDateTime("2016-09-26T09:30:00.0Z") # period end

        fig = plot_Time_Series(doi, doi_end, files_path, filter=filter,
                               bandpass=bandpass)

    """

    # find all files for specified day
    day_file_list = sorted(glob.glob(f"{files_path}/*.{doi.year}"
                                     f"-{doi.month:02}-{doi.day:02}.ms"))

    # load files into stream
    st = Stream()
    for file in day_file_list:
        st += read(file)

        # filter before trimming to avoid edge effects
        if filter:
            # bandpass filter specified frequencies
            st.filter('bandpass', freqmin=bandpass[0], freqmax=bandpass[1])

        st.trim(doi - 30, doi_end + 30)

    # initialize figure and set the figure size
    figureWidth = 50  # 80
    figureHeight = 3.5 * len(st)  # 0.6 for all stations # 3.5
    fig = plt.figure(figsize=(figureWidth, figureHeight))
    amplitude_plot = fig.add_subplot()
    # make a subplot of subplots for spectrograms
    # amplitude_plot = plt.subplot(2, 1, 1)

    # loop through stream and generate plots
    y_labels = []
    for index, trace in enumerate(st):
        # find max trace value for normalization
        maxTraceValue, _ = max_amplitude(trace)

        # define data to work with
        time = trace.times("matplotlib")
        norm_amplitude = (trace.data - np.min(trace.data)) / (maxTraceValue -
                                                              np.min(
                                                                  trace.data)) * 1.25 + index
        # add trace to waveform plot
        amplitude_plot.plot_date(time, norm_amplitude, fmt="k-", linewidth=0.7)

        # plot time markers for this trace if they exist
        network_station = f"{trace.stats.network}.{trace.stats.station}"
        print(
            f"{network_station} in station dict:{network_station in time_markers}")
        if network_station in time_markers:
            time_marker = time_markers[network_station]
            # plot time_marker box
            x_vals = [time_marker[0].matplotlib_date,
                      time_marker[0].matplotlib_date,
                      time_marker[1].matplotlib_date,
                      time_marker[1].matplotlib_date,
                      time_marker[0].matplotlib_date]
            y_vals = [min(norm_amplitude), max(norm_amplitude),
                      max(norm_amplitude), min(norm_amplitude),
                      min(norm_amplitude)]
            amplitude_plot.plot_date(x_vals, y_vals,
                                     fmt="r-", linewidth=2.0)

        # add station name to list of y labels
        y_labels.append(f"{network_station}.{trace.stats.channel}")

    # print(trace.stats.sampling_rate)

    # set axes attributes
    amplitude_plot.set_yticks(np.arange(0.5, len(st) + 0.5))
    amplitude_plot.set_yticklabels(y_labels)
    amplitude_plot.set_ylabel('Station.Channel')
    amplitude_plot.set_xlim([doi.matplotlib_date, doi_end.matplotlib_date])
    amplitude_plot.set_xlabel(f'Time: Hr:Min:Sec of {doi.month:02}-'
                              f'{doi.day:02}-{doi.year}')
    myFmt = DateFormatter("%H:%M:%S")  # "%H:%M:%S.f"
    amplitude_plot.xaxis.set_major_formatter(myFmt)
    locator_x = AutoDateLocator(minticks=10, maxticks=35)
    amplitude_plot.xaxis.set_major_locator(locator_x)
    amplitude_plot.set_ylim((0, len(st) + 0.5))
    # fig.tight_layout()
    fig.savefig(f"{doi.month:02}-{doi.day:02}-{doi.year}T{doi.hour:02}."
                f"{doi.minute:02}.png", dpi=100)

    plt.show()

    return fig


# plot detections from matched-filter Party object detections and
# corresponding data files
def plot_party_detections(party, detection_files_path, filter=True,
                          title=False, save=False):
    """
        Plots time series for all detections in a Party object from files in
        the `detection_files_path` directory. If there are more than 100
        detections, only the first 150 detections are plotted.

        Example:
            # load party object from file
            # infile = open('party_09_26_2016_to_10_02_2016_abs.25.pkl', 'rb')
            infile = open('party_06_15_2016_to_08_12_2018_abs.25.pkl', 'rb')
            party = pickle.load(infile)
            infile.close()

            # define path of files from detection
            detection_files_path = "/Users/human/ak_data/inner"

            # set snr threshold and cull the party detections
            from lfe_finder import cull_detections
            snr_threshold = 3.5
            culled_party = cull_detections(party, detection_files_path, snr_threshold)

            # plot the party detections and a histogram of the SNRs
            title = "snr3.5_abs_0.25_detections"
            fig = plot_party_detections(culled_party,
                                        detection_files_path,
                                        title=title, save=True)

    """
    # initialize figure and set the figure size
    figureWidth = 12
    figureHeight = 0.5 * len(party.families[0].detections)
    fig = plt.figure(figsize=(figureWidth, figureHeight))
    amplitude_plot = fig.add_subplot()

    # check number of detections and trim if necessary
    if len(party.families[0].detections) > 60:
        party.families[0].detections = party.families[0].detections[:60]

    y_labels = [] # keep track of y-axis labels (station and channel names)
    snrs = []
    # loop over detections
    for index, detection in enumerate(party.families[0].detections):
        # define the date of interest to get appropriate files
        doi = detection.detect_time

        # find all files for specified day
        day_file_list = sorted(glob.glob(f"{detection_files_path}/TA.N25K.BHZ"
                                         f".{doi.year}-{doi.month:02}"
                                         f"-{doi.day:02}.ms"))

        # should only be one file, guard against many
        file = day_file_list[0]

        # load file into stream
        st = Stream()
        st += read(file)

        # filter before trimming to avoid edge effects
        if filter:
            # bandpass filter specified frequencies
            st.filter('bandpass', freqmin=1, freqmax=15)

        st.trim(doi - 0.5, doi + 16)
        trace = st[0] # get the trace
        snrs.append(snr(trace)[0])

        if index == 0:
            time = trace.times("matplotlib")
            plot_start = trace.stats.starttime
            plot_end = trace.stats.endtime

        # find max trace value for normalization
        maxTraceValue, _ = max_amplitude(trace)

        # define data to work with
        norm_amplitude = (trace.data - np.min(trace.data)) / (maxTraceValue -
                          np.min(trace.data)) * 1.25 + index
        # add trace to waveform plot
        amplitude_plot.plot_date(time, norm_amplitude, fmt="k-", linewidth=0.7)

        # plot time markers for this trace if they exist
        network_station = f"{trace.stats.network}.{trace.stats.station}"

        # add station name to list of y labels
        y_labels.append(f"{network_station}.{trace.stats.channel}")

    # set axes attributes
    amplitude_plot.set_yticks(np.arange(0.5, len(party.families[0].detections) + 0.5))
    amplitude_plot.set_yticklabels(y_labels)
    amplitude_plot.set_ylabel('Station.Channel')
    amplitude_plot.set_xlim([plot_start.matplotlib_date,
                             plot_end.matplotlib_date])
    amplitude_plot.set_xlabel(f'Time: Hr:Min:Sec')
    myFmt = DateFormatter("%H:%M:%S")  # "%H:%M:%S.f"
    amplitude_plot.xaxis.set_major_formatter(myFmt)
    locator_x = AutoDateLocator(minticks=8, maxticks=35)
    amplitude_plot.xaxis.set_major_locator(locator_x)
    amplitude_plot.set_ylim((0, len(party.families[0].detections) + 0.5))
    # fig.tight_layout()

    if title != False:
        amplitude_plot.set_title(title)
        if save:
            fig.savefig(f"{title}.png", dpi=100)

    plt.show()

    # # plot the detections snr distribution
    # hist = plot_distribution(snrs, title="Detections SNR distribution",
    #                          save=save)

    return fig #, hist


# plot stack of waveforms on common time axis
def plot_stack(stack, filter=False, bandpass=[], title=False, save=False):
    """
    Plots time series for all traces in a stack on a common time axis. Data
    are first bandpass filtered if `filter=True`.

    Example:
        # load stream list from file
        import pickle
        infile = open('stack_list.pkl', 'rb')
        stack_list = pickle.load(infile)
        infile.close()

        # bandpass filter from 1-15 Hz
        filter = False
        bandpass = [1, 15]

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
    # # check for infinitys returned by EQcorrscan stack
    # for trace in stack:
    # 	if trace.data[0] == np.inf:
    # 		stack.remove(trace)

    # filtering will have edge effects
    if filter:
        # bandpass filter specified frequencies
        stack.filter('bandpass', freqmin=bandpass[0], freqmax=bandpass[1])

    # initialize figure and set the figure size
    figureWidth = 20  # 80
    figureHeight = 2.5 * len(stack)  # 0.6 for all stations # 3.5
    fig = plt.figure(figsize=(figureWidth, figureHeight))
    amplitude_plot = fig.add_subplot()

    # loop through stream and generate plots
    y_labels = []
    # set common time axis
    time = stack[0].times("matplotlib")
    for index, trace in enumerate(stack):
        # find max trace value for normalization
        maxTraceValue, _ = max_amplitude(trace)

        # define data to work with
        norm_amplitude = (trace.data - min(trace.data)) / (maxTraceValue -
                         min(trace.data)) * 1.25 + index

        # add trace to waveform plot
        amplitude_plot.plot_date(time, norm_amplitude, fmt="k-", linewidth=0.7)

        # plot time markers for this trace if they exist
        network_station = f"{trace.stats.network}.{trace.stats.station}"

        # add station name to list of y labels
        y_labels.append(f"{network_station}.{trace.stats.channel}.{index}")

    # set axes attributes
    amplitude_plot.set_yticks(np.arange(0.5, len(stack) + 0.5))
    amplitude_plot.set_yticklabels(y_labels)
    amplitude_plot.set_ylabel('Station.Channel')
    amplitude_plot.set_xlim([time[0], time[-1]])
    amplitude_plot.set_xlabel(f'Time: Hr:Min:Sec')
    myFmt = DateFormatter("%H:%M:%S")  # "%H:%M:%S.f"
    amplitude_plot.xaxis.set_major_formatter(myFmt)
    locator_x = AutoDateLocator(minticks=10, maxticks=35)
    amplitude_plot.xaxis.set_major_locator(locator_x)
    amplitude_plot.set_ylim((0, len(stack) + 0.5))
    if title != False:
        amplitude_plot.set_title(title)
        if save:
            fig.savefig(f"{title}.png", dpi=100)
    fig.tight_layout()
    plt.show()

    return fig


def plot_template_and_stack(party, stack_lin, stack_pw,
                            detection_files_path, template, save=False,
                            title=False):
    """Plots all templates in a party, with waveform files stored in the
    specified path, and the corresponding linear and phase weighted stacks."""
    family = sorted(party.families, key=lambda f: len(f))[-1]

    # get template stream
    family_stream = family.template.st

    template_stack = Stream()
    stack_len = stack_lin[0].stats.endtime - stack_lin[
        0].stats.starttime
    stack_start = stack_lin[0].stats.starttime
    # assemble templates and stacks into a single stream for plotting
    for trace in family_stream:
        # get template info
        t_net = trace.stats.network
        t_sta = trace.stats.station
        t_cha = trace.stats.channel
        t_date = trace.stats.starttime

        # get file containing template time series
        file_list = glob.glob(f"{detection_files_path}/{t_net}."
                              f"{t_sta}.{t_cha}.{t_date.year}"
                              f"-{t_date.month:02}-{t_date.day:02}.ms")
        # guard against missing files
        if len(file_list) > 0:
            # should only be one file, but safeguard against many
            file = file_list[0]
            # load day file into stream
            template_trace = read(file)
            template_trace.filter('bandpass', freqmin=1, freqmax=15)
            # trim trace to time surrounding pick time
            if template == 1:
                template_start = UTCDateTime("2016-09-26T09:28:21.5Z")
            elif template == 2:
                template_start = UTCDateTime("2016-09-27T07:37:32.0Z")
            elif template == 4:
                template_start = UTCDateTime("2016-09-27T06:30:55.5Z")
            elif template == 5:
                template_start = UTCDateTime("2016-09-26T09:25:34.5Z")

            # make sure sampling is 40 Hz
            template_trace.interpolate(sampling_rate=40.0)

            template_trace.trim(template_start, template_start +
                                stack_len, pad=True, fill_value=0,
                                nearest_sample=True)

            # go from stream to trace
            template_trace = template_trace[0]

        # get trace from linear stack and phase weighted stack with
        #  corresponding net/sta/chan
        lin_trace = stack_lin.select(network=t_net, station=t_sta,
                                     channel=t_cha).copy()[0]
        lin_trace.stats.station += "_LIN_STACK"
        pw_trace = stack_pw.select(network=t_net, station=t_sta,
                                   channel=t_cha).copy()[0]
        pw_trace.stats.station += "_PW_STACK"

        # add template tag to templates
        template_trace.stats.station += "_TEMPLATE"
        template_stack += template_trace
        template_stack += lin_trace
        template_stack += pw_trace

    fig = plot_stack(template_stack, title=title, save=save, filter=False)

    return fig


def plot_stream_relative():
    pass


def plot_stream_absolute(stream, title=False, save=False, figWidth=False):
    """ Plots all traces of a stream in absolute time. This function is
    useful for visualizing shifted traces of varying length."""
    # initialize figure and set the figure size
    figureWidth = 10
    if figWidth != False:
        figureWidth = figureWidth * figWidth  # figWidth is (0,1]

    figureHeight = 0.5 * len(stream)  # 0.6 for all stations # 3.5
    fig = plt.figure(figsize=(figureWidth, figureHeight))
    amplitude_plot = fig.add_subplot()

    # get time period of data for plotting on common time axis
    for tr_idx, trace in enumerate(stream):
        # set some figure parameters to update from the first trace
        if tr_idx == 0:
            # initialize figure time period to update
            earliest_time = trace.stats.starttime
            latest_time = trace.stats.endtime

        else:
            # update figure time period
            if trace.stats.starttime < earliest_time:
                earliest_time = trace.stats.starttime
            if trace.stats.endtime > latest_time:
                latest_time = trace.stats.endtime

    # set common time axis from scanned data
    timeTrace = Trace()
    timeTrace.stats.sampling_rate = 40
    timeTrace.stats.starttime = earliest_time
    timeTrace.stats.npts = (
                                   latest_time - earliest_time) * 40  # * sampling rate
    time = timeTrace.times("matplotlib")

    # loop through stream and generate plots
    y_labels = []
    for tr_idx, trace in enumerate(stream):
        # find max trace value for normalization
        maxTraceValue, _ = max_amplitude(trace)

        # define data to work with
        norm_amplitude = (trace.data - np.min(trace.data)) / (
                maxTraceValue - np.min(trace.data)) * 1.25 + tr_idx
        # add trace to waveform plot
        amplitude_plot.plot_date(trace.times("matplotlib"), norm_amplitude,
                                 fmt="k-", linewidth=0.7)

        # plot time markers for this trace if they exist
        network_station = f"{trace.stats.network}.{trace.stats.station}"

        # add station name to list of y labels
        y_labels.append(f"{network_station}.{trace.stats.channel}")

    # set plot attributes
    amplitude_plot.set_yticks(np.arange(0.5, len(stream) + 0.5))
    amplitude_plot.set_yticklabels(y_labels)
    amplitude_plot.set_ylabel('Station.Channel')
    amplitude_plot.set_xlim([time[0], time[-1]])
    amplitude_plot.set_xlabel(f'Time: Hr:Min:Sec')
    myFmt = DateFormatter("%H:%M:%S")  # "%H:%M:%S.f"
    amplitude_plot.xaxis.set_major_formatter(myFmt)
    locator_x = AutoDateLocator(minticks=12, maxticks=30)
    amplitude_plot.xaxis.set_major_locator(locator_x)
    amplitude_plot.set_ylim((0, len(stream) + 0.5))
    fig.tight_layout()
    if title != False:
        amplitude_plot.set_title(title)
        if save:
            fig.savefig(f"{title}.png", dpi=100)

    plt.show()

    return fig


def plot_distribution(data, bins=False, title=False, save=False):
    """ Generates a histogram from the specified data.

    Args:
        data: list of floats or ints

    Returns:
        None

    Example:
        plot_distribution(snrs, title="SNR distribution", save=True)
    """
    df = pd.Series(data)
    fig = plt.figure(figsize=(9, 5))
    if bins == False:
        bins = 100
    n, bins, patches = plt.hist(data, bins=bins, facecolor="darkred",
                                alpha=0.6)
    ax = plt.axes()
    # set background color
    ax.set_facecolor("dimgrey")
    # set plot labels
    plt.xlabel(f'SNR (100 bins)')
    plt.ylabel("Counts per bin")
    # set plot limits
    # plt.ylim(0, 50)
    ax.set_xlim([bins[0], bins[-1]])

    plt.grid(True)

    # plot kde and quantiles on a different y axis
    ax2 = ax.twinx()
    # plot kde
    df.plot(kind="kde", ax=ax2)
    # quantile lines
    quant_5, quant_25, quant_50, quant_75, quant_95 = df.quantile(0.05), \
                                                      df.quantile(0.25), \
                                                      df.quantile(0.5), \
                                                      df.quantile(0.75), \
                                                      df.quantile(0.95)
    quants = [[quant_5, 0.6, 0.16], [quant_25, 0.8, 0.26], [quant_50, 1, 0.36],
              [quant_75, 0.8, 0.46], [quant_95, 0.6, 0.56]]
    for i in quants:
        ax2.axvline(i[0], alpha=i[1], ymax=i[2], linestyle=":", c="k")

    # set ax2 y labels and ticks
    ax2.set_ylim(0, 1)
    ax2.set_yticklabels([])
    ax2.set_ylabel("")

    # quantilennotations
    ax2.text(quant_5 - .1, 0.17, "5th", size=10, alpha=0.8)
    ax2.text(quant_25 - .13, 0.27, "25th", size=11, alpha=0.85)
    ax2.text(quant_50 - .13, 0.37, "50th", size=12, alpha=1)
    ax2.text(quant_75 - .13, 0.47, "75th", size=11, alpha=0.85)
    ax2.text(quant_95 - .25, 0.57, "95th Percentile", size=10, alpha=.8)

    if title != False:
        ax.set_title(title, y=0.9999)
        if save:
            fig.savefig(f"{title}.png", dpi=100)

    plt.tight_layout()
    plt.show()

    return fig


def examine_stack():
    "Animate building of stack"

    ...