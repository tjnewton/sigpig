"""
Functions to generate various figures. 
"""

import obspy
from obspy import read, Stream
from obspy.core.utcdatetime import UTCDateTime
from obspy.clients.fdsn import Client
import calendar
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
									 bandpass=[]):
	"""
	Plots time series and spectrograms for all files in the `files_path` for
	the specified time period starting at `doi` and ending at `doi_end`. Data
	are first bandpass filtered if `filter=True`.

	Example:
		# define dates of interest
		doi = UTCDateTime("2016-09-26T09:24:53.0Z") # period start
		doi_end = UTCDateTime("2016-09-26T09:26:53.0Z") # period end

		# define time series files path
		files_path = "./subset_stations"

		# bandpass filter from 2-8 Hz
		bandpass = [2, 8]

		fig = plot_Time_Series_And_Spectrogram(doi, doi_end, files_path,
											   filter=True, bandpass=bandpass)

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
	figureWidth = 50
	figureHeight = 3.5 * len(st) # 0.6 for all stations
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
		# add station name to list of y labels
		y_labels.append(f"{trace.stats.network}.{trace.stats.station}"
						f".{trace.stats.channel}")

		print(trace.stats.sampling_rate)

		# build information for spectrogram
		duration = doi_end - doi
		num_windows = 80
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
		spec.pcolormesh(tSTFT, fSTFT, 20 * np.log10(np.absolute(STFT)))
		spec.set_xlim([0, duration - window_length /trace.stats.sampling_rate])
		spec.set_ylabel(f"{trace.stats.network}.{trace.stats.station}."
						f"{trace.stats.channel}",
						rotation=0, labelpad=40)
		spec.tick_params(axis='x', which='both', bottom=False, top=False,
						 labelbottom=False)
		spec.set_yticks([])

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