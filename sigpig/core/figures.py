"""
Functions to generate various figures. 
"""

import obspy
from obspy import read, Stream
from obspy.core.utcdatetime import UTCDateTime
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.dates import DateFormatter, AutoDateLocator
import glob
import numpy as np
import scipy.signal as spsig

#FIXME ::: Add elevation functions from Desktop/autopicker/preprocessing/station_locations.py
# above requires configuring environment with rasterio and dependencies


def rattlesnake_Ridge_Station_Locations():
    """ Returns a dict of station locations, used by EQTransformer
    downloader.stationListFromMseed to create a station_list.json file"""
    stations = (1,2,3,4,5,6,7,8,9,10,12,13,15,16,17,18,20,21,22,23,25,26,27,
                28,30,31,32,33,34,35,36,37,38,39,40,41,42,'UGAP3','UGAP5',
                'UGAP6')
    latitudes = (46.5283582,46.5282788,46.5282528,46.5281667,46.5281717,
                 46.5281396,46.5281154,46.5280997,46.5280294,46.5279883,
                 46.527869,46.5280507,46.5277918,46.5276182,46.5275217,
                 46.5273933,46.5272481,46.5271363,46.5270128,46.5269864,
                 46.5268652,46.5266988,46.5266946,46.5265515,46.5264383,
                 46.5263249,46.526291,46.5261454,46.526008,46.5260073,
                 46.5257922,46.5257573,46.5255906,46.525457,46.5253461,
                 46.5253264,46.5251281,46.52764,46.52663,46.52662)
    longitudes = (-120.4667914,-120.4662543,-120.4671993,-120.4663367,
                  -120.4665884,-120.4658735,-120.4668433,-120.4661669,
                  -120.4669906,-120.4660337,-120.4660045,-120.4663792,
                  -120.4662038,-120.4658907,-120.4658828,-120.4657754,
                  -120.465704,-120.4656657,-120.4652239,-120.4656266,
                  -120.4656232,-120.465502,-120.4652094,-120.4655297,
                  -120.4655254,-120.4651244,-120.465529,-120.4655384,
                  -120.4650622,-120.4654893,-120.465437,-120.4649769,
                  -120.4653985,-120.4654196,-120.4650617,-120.4654122,
                  -120.4654584,-120.46564,-120.46553,-120.46637)
    elevations = (435.5141526063508,459.94244464035376,419.17496055551493,
                  456.4002279560358,445.2225853337437,471.96515712383257,
                  433.75775906071993,463.0207557157984,427.2243674889657,
                  466.22453723535386,465.71123190316104,454.23827459858546,
                  459.72188069932287,464.2974124921104,462.3340642200634,
                  459.20987810605817,455.81267908165233,452.4536132975136,
                  449.39541344789984,447.7972434217686,444.0736248417953,
                  440.47945136171927,441.41787035305947,436.52746672411286,
                  433.50566973487213,433.6467649166143,430.2554423791601,
                  425.63420243981653,426.1603861687825,421.53271734479665,
                  416.2025875386629,421.73271284375056,411.9060515678977,
                  407.8851035175373,412.7486362327675,404.519552729108,
                  397.5652740413809,465.2,435.4,435.4)
    location_dict = {}
    for index, station in enumerate(stations):
        location_dict[str(station)] = [latitudes[index], longitudes[index],
                                       elevations[index]]
    return location_dict

def eqTransformer_Formatter(project_Name: str, start_Time, end_Time):
    """
    Formats data into EQTransformer station subdirectory format and naming
    convention.

    Example:
    start_Time = UTCDateTime("2018-03-13T01:31:59.0Z")
    end_Time =   UTCDateTime("2018-03-13T01:41:01.0Z")
    project_Name = "Rattlesnake Ridge"
    eqTransformer_Formatter(project_Name, start_Time, end_Time)

    """

    if project_Name == "Rattlesnake Ridge":
        # build filepath list based on dates
        node = ['DP1', 'DP2', 'DPZ'] # nodal seismometer channels
        ugap = ['EHN', 'EHE', 'EHZ']
        stations_channels = {'1': node, '2': node, '3': node, '4': node,
                             '5': node, '6': node, '7': node, '8': node,
                             '9': node, '10': node, '12': node, '13': node,
                             '15': node, '16': node, '17': node, '18': node,
                             '20': node, '21': node, '22': node, '23': node,
                             '25': node, '26': node, '27': node, '28': node,
                             '30': node, '31': node, '32': node, '33': node,
                             '34': node, '35': node, '36': node, '37': node,
                             '38': node, '39': node, '40': node, '41': node,
                             '42': node, 'UGAP3': ugap, 'UGAP5': ugap, 'UGAP6': ugap}

        for station in stations_channels:
            print(f" Processing {station}")
            # make directory based on station name
            os.mkdir(f"/Users/human/Dropbox/Programs/eqt/rr/downloads_mseeds/{station}")

            for channel in stations_channels[station]:
                filepath = f"/Volumes/newton_6TB/RR_MSEED/5A.{station}.." \
                           f"{channel}.{start_Time.year}-" \
                           f"{start_Time.month:02}-{start_Time.day:02}T00.00.00.ms"
                # load data, trim, and save file
                #obspyStream = Stream()
                obspyStream = read(filepath)

                interpolate = False
                # # interpolation breaks on UGAP6
                # for trace in obspyStream:
                #     # the sampling rate of nodes is 250 Hz
                #     if trace.stats.sampling_rate != 250.0:
                #         print(f"Trace has a different sampling rate than "
                #               f"nodes. ")
                #         # raise the flag
                #         interpolate = True
                if interpolate:
                    print("Interpolating...")
                    # interpolate to correct sampling rate and trim to correct time period
                    sampling_Rate = 250.0
                    npts = int((end_Time - start_Time) * sampling_Rate)
                    obspyStream.interpolate(sampling_Rate, method="lanczos",
                                            starttime=start_Time, npts=npts,
                                            a=30)
                    # obspy.signal.interpolation.plot_lanczos_windows(a=30)
                else:
                    # trim to specified time period
                    obspyStream = obspyStream.trim(start_Time, end_Time)

                # format filename and save Stream as miniseed file
                network = obspyStream[0].stats.network


                start_Time_Stamp = f"{obspyStream[0].stats.starttime.year}{obspyStream[0].stats.starttime.month:02}{obspyStream[0].stats.starttime.day:02}T{obspyStream[0].stats.starttime.hour:02}{obspyStream[0].stats.starttime.minute:02}{obspyStream[0].stats.starttime.second:02}Z"
                end_Time_Stamp = f"{obspyStream[0].stats.endtime.year}{obspyStream[0].stats.endtime.month:02}{obspyStream[0].stats.endtime.day:02}T{obspyStream[0].stats.endtime.hour:02}{obspyStream[0].stats.endtime.minute:02}{obspyStream[0].stats.endtime.second:02}Z"

                # writes to path
                obspyStream.write(f"/Users/human/Dropbox/Programs/eqt"
                                  f"/rr/downloads_mseeds/{station}/"
                                  f"{network}.{station}.."
                                  f"{channel}__{start_Time_Stamp}__"
                                  f"{end_Time_Stamp}.mseed", format="MSEED")

    return None


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
    maxTraceValue = max_Amplitude(subset_data)
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

# helper function to find max amplitude of a time series for plotting
def max_Amplitude(timeSeries):
	'''
	(np.ndarray or obspy stream or trace object) -> (float)

	Determines single max value that occurs in a numpy array or trace or in
	all traces of a stream.
	'''
	# for Stream objects
	if isinstance(timeSeries, obspy.core.stream.Stream):
		# creates a list of the max value in each trace
		traceMax = [np.max(np.abs(timeSeries[trace].data)) for trace in
					range(len(timeSeries))]
		# return max value among all traces
		return np.max(traceMax)

	# for Trace objects
	elif isinstance(timeSeries, obspy.core.trace.Trace):
		traceMax = np.max(np.abs(timeSeries.data))
		# return max value
		return traceMax

	elif isinstance(timeSeries, np.ndarray):
		# returns the max for each row, works with 1D and 2D np.ndarrays
		if len(timeSeries.shape) == 1:  # 1D case
			return np.abs(timeSeries).max()
		elif len(timeSeries.shape) == 2:  # 2D case
			return np.abs(timeSeries).max(1)
		else:
			print("You broke the max_Amplitude function with a np.ndarray "
				  "that is not 1D or 2D ;(")

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
	day_file_list = sorted(glob.glob(f"{files_path}/*Z.{doi.year}"
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
	figureHeight = 3.5 * len(st) # 0.6 for all stations # 3.5
	fig = plt.figure(figsize=(figureWidth, figureHeight))
	gs = fig.add_gridspec(3, 1)
	amplitude_plot = fig.add_subplot(gs[0,:])
	frequency_plot = fig.add_subplot(gs[1:,:])
	# make a subplot of subplots for spectrograms
	frequency_plots = gridspec.GridSpecFromSubplotSpec(len(st), 1,
												   subplot_spec=frequency_plot)
	# amplitude_plot = plt.subplot(2, 1, 1)
	# frequency_plot = plt.subplot(2, 1, 2)

	# loop through stream and generate plots
	y_labels = []
	for index, trace in enumerate(st):
		# find max trace value for normalization
		maxTraceValue = max_Amplitude(trace)

		# define data to work with
		time = trace.times("matplotlib")
		trace_start = trace.stats.starttime
		norm_amplitude = (trace.data - np.min(trace.data)) / (maxTraceValue -
						  np.min(trace.data)) * 1.25 + index
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
		num_windows = (duration/60) * 40 - 2
		window_duration = duration / num_windows
		window_length = int(window_duration * trace.stats.sampling_rate)
		nfftSTFT = window_length * 2 # nyquist
		overlap = int(window_length / 2) # overlap between spec. windows in samples
		[fSTFT, tSTFT, STFT] = spsig.spectrogram(trace.data,
												 trace.stats.sampling_rate,
												 nperseg=window_length,
											   noverlap=overlap, nfft=nfftSTFT)
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
	amplitude_plot.set_yticks(np.arange(0.5, len(st)+0.5))
	amplitude_plot.set_yticklabels(y_labels)
	amplitude_plot.set_ylabel('Station.Channel')
	amplitude_plot.set_xlim([doi.matplotlib_date, doi_end.matplotlib_date])
	amplitude_plot.set_xlabel(f'Time: Hr:Min:Sec of {doi.month:02}-'
							  f'{doi.day:02}-{doi.year}')
	myFmt = DateFormatter("%H:%M:%S")  # "%H:%M:%S.f"
	amplitude_plot.xaxis.set_major_formatter(myFmt)
	locator_x = AutoDateLocator(minticks=10, maxticks=35)
	amplitude_plot.xaxis.set_major_locator(locator_x)
	amplitude_plot.set_ylim((0, len(st)+0.5))
	# frequency_plot.set_ylabel('Frequency (Hz)')
	# frequency_plot.set_xlabel('Time (s)')
	frequency_plot.set_yticks([])
	frequency_plot.set_xticks([])
	# fig.tight_layout()
	fig.savefig(f"{doi.month:02}-{doi.day:02}-{doi.year}T{doi.hour:02}."
				f"{doi.minute:02}.png", dpi=100)

	plt.show()


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
	figureWidth = 50 # 80
	figureHeight = 0.5 * len(st) # 0.6 for all stations # 3.5
	fig = plt.figure(figsize=(figureWidth, figureHeight))
	amplitude_plot = fig.add_subplot()
	# make a subplot of subplots for spectrograms
	# amplitude_plot = plt.subplot(2, 1, 1)

	# loop through stream and generate plots
	y_labels = []
	for index, trace in enumerate(st):
		# find max trace value for normalization
		maxTraceValue = max_Amplitude(trace)

		# define data to work with
		time = trace.times("matplotlib")
		norm_amplitude = (trace.data - np.min(trace.data)) / (maxTraceValue -
						  np.min(trace.data)) * 1.25 + index
		# add trace to waveform plot
		amplitude_plot.plot_date(time, norm_amplitude, fmt="k-", linewidth=0.7)

		# plot time markers for this trace if they exist
		network_station = f"{trace.stats.network}.{trace.stats.station}"
		print(f"{network_station} in station dict:{network_station in time_markers}")
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
	amplitude_plot.set_yticks(np.arange(0.5, len(st)+0.5))
	amplitude_plot.set_yticklabels(y_labels)
	amplitude_plot.set_ylabel('Station.Channel')
	amplitude_plot.set_xlim([doi.matplotlib_date, doi_end.matplotlib_date])
	amplitude_plot.set_xlabel(f'Time: Hr:Min:Sec of {doi.month:02}-'
							  f'{doi.day:02}-{doi.year}')
	myFmt = DateFormatter("%H:%M:%S")  # "%H:%M:%S.f"
	amplitude_plot.xaxis.set_major_formatter(myFmt)
	locator_x = AutoDateLocator(minticks=10, maxticks=35)
	amplitude_plot.xaxis.set_major_locator(locator_x)
	amplitude_plot.set_ylim((0, len(st)+0.5))
	# fig.tight_layout()
	fig.savefig(f"{doi.month:02}-{doi.day:02}-{doi.year}T{doi.hour:02}."
				f"{doi.minute:02}.png", dpi=100)

	plt.show()