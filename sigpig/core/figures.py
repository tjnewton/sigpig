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
		doi = UTCDateTime("2016-09-26T09:10:53.0Z") # period start
		doi_end = UTCDateTime("2016-09-26T09:40:53.0Z") # period end

		# define time series files path
		files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/picked"

		# bandpass filter from 2-8 Hz
		filter = False
		bandpass = [2, 8]

		# plot time markers on specified stations
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

	"""

	# find all files for specified day
	day_file_list = glob.glob(f"{files_path}/*.BHZ.{doi.year}-{doi.month:02}"
							  f"-{doi.day:02}.ms")
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
									 fmt="r-", linewidth=1.0)

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
						cmap='inferno')
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
				f"{doi.minute:02}.png", dpi=300)

	plt.show()

def plot_Time_Series(doi, doi_end, files_path, filter=False, bandpass=[],
					 time_markers={}):
	"""
	Plots time series for all files in the `files_path` for
	the specified time period starting at `doi` and ending at `doi_end`. Data
	are first bandpass filtered if `filter=True`.

	Example:
		# define dates of interest
		doi = UTCDateTime("2016-09-26T09:25:30.0Z") # period start
		doi_end = UTCDateTime("2016-09-26T09:26:30.0Z") # period end

		# define time series files path
		files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/picked"

		# bandpass filter from 2-8 Hz
		filter = True
		bandpass = [2, 8]

		# plot time markers on specified stations
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

		fig = plot_Time_Series(doi, doi_end, files_path, filter=filter,
							   bandpass=bandpass, time_markers=time_markers)

		# plot different date range without time markers
		doi = UTCDateTime("2016-09-26T09:28:00.0Z") # period start
		doi_end = UTCDateTime("2016-09-26T09:30:00.0Z") # period end

		fig = plot_Time_Series(doi, doi_end, files_path, filter=filter,
							   bandpass=bandpass)

	"""

	# find all files for specified day
	day_file_list = glob.glob(f"{files_path}/*.{doi.year}-{doi.month:02}"
							  f"-{doi.day:02}.ms")

	# load files into stream
	st = Stream()
	for file in day_file_list:
		st += read(file)

		# filter before trimming to avoid edge effects
		if filter:
			# bandpass filter specified frequencies
			st.filter('bandpass', freqmin=bandpass[0], freqmax=bandpass[1])

		st.trim(doi, doi_end)

	# initialize figure and set the figure size
	figureWidth = 50 # 80
	figureHeight = 1.5 * len(st) # 0.6 for all stations # 3.5
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
									 fmt="r-", linewidth=1.0)

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
				f"{doi.minute:02}.png", dpi=300)

	plt.show()