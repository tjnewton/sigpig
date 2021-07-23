# sigpig
## Seismology and Signal Processing Toolkit 

### To get started:
Clone this repo in your preferred directory:  
`git clone https://github.com/tjnewton/sigpig.git`  
Move into the sigpig directory:  
`cd sigpig`  
Use conda to create a Python environment from the sigpig.yml file:  
`conda env create -f sigpig.yml`  
Activate the environment:  
`conda activate sigpig`  
Launch Jupyter Lab within snippets environment:  
`jupyter lab`  
Browse the repo directories and notebooks within Jupyter Lab :)

## Functionality:
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
doi = UTCDateTime("2016-09-26T09:28:00.0Z") # period start
doi_end = UTCDateTime("2016-09-26T09:30:00.0Z") # period end

# define time series files path
files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/picked"

# bandpass filter from 1-15 Hz
filter = True
bandpass = [1, 15]

# plot time markers on specified stations
time_markers = {"AK.GLB":  [UTCDateTime("2016-09-26T09:28:40.0Z"),
			    UTCDateTime("2016-09-26T09:28:54.0Z")],
		"AK.PTPK": [UTCDateTime("2016-09-26T09:28:55.0Z"),
			    UTCDateTime("2016-09-26T09:29:09.0Z")],
		"AV.WASW": [UTCDateTime("2016-09-26T09:28:36.0Z"),
			    UTCDateTime("2016-09-26T09:28:50.0Z")]}

fig = plot_Time_Series_And_Spectrogram(doi, doi_end, files_path,
				       filter=filter, bandpass=bandpass,
				       time_markers=time_markers)
```
![](doc/images/ts-spect.png?raw=true)

Time series only plotting  
```
fig = plot_Time_Series(doi, doi_end, files_path, filter=filter,
		       bandpass=bandpass)
```
![](doc/images/ts.png?raw=true)
