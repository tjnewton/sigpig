# sigpig
Seismology and Signal Processing Toolkit 

### sigpig.core.data downloads data:  
Time series data from IRIS DMC  
```
network = "AK"
stations = ["BAL", "BARN", "DHY", "DIV", "DOT", "GHO", "GLB", "K218",
            "KLU", "KNK", "MCAR", "MCK", "PAX", "PTPK", "RIDG", "RND",
            "SAW", "SCM", "VRDI", "WAT1", "WAT6", "WAT7"]
location = "**"
channels = ["BHZ", "BNZ", "HNZ"]
start_Time = UTCDateTime("2016-06-15T00:00:00.0Z")
end_Time =   UTCDateTime("2017-08-11T23:59:59.999999999999999Z")
get_Waveforms(network, stations, location, channels, start_Time, end_Time)
```

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