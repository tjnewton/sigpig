# sigpig
Seismology and Signal Processing Toolkit 

### sigpig.core.figures generates various figures:  
Time series and spectrogram plotting  
```
# define dates of interest
doi = UTCDateTime("2016-09-26T09:24:53.0Z") # period start
doi_end = UTCDateTime("2016-09-26T09:26:53.0Z") # period end

# define time series files path
files_path = "files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/subset_stations"

# bandpass filter from 1-10 Hz
bandpass = [1, 10]

# plot time markers on specified stations
time_markers = {"AV.WASW": [UTCDateTime("2016-09-26T09:25:49.0Z"),
					 		UTCDateTime("2016-09-26T09:26:03.0Z")],
				"TA.N25K": [UTCDateTime("2016-09-26T09:25:52.5Z"),
					 		UTCDateTime("2016-09-26T09:26:06.5Z")],
				"YG.MCR3": [UTCDateTime("2016-09-26T09:25:52.0Z"),
					 		UTCDateTime("2016-09-26T09:26:06.0Z")]}

fig = plot_Time_Series_And_Spectrogram(doi, doi_end, files_path, filter=True, bandpass=bandpass, time_markers=time_markers)
```
![](doc/images/ts-spect.png?raw=true)
