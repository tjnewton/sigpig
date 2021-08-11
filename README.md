# sigpig
## Seismology and Signal Processing Toolkit 
built on top of [matplotlib](https://github.com/matplotlib/matplotlib), [ObsPy](https://github.com/obspy/obspy), [GDAL](https://github.com/OSGeo/gdal), [Laspy](https://github.com/laspy/laspy), [NumPy](https://github.com/numpy/numpy), [pptk](https://github.com/heremaps/pptk), and [EQcorrscan](https://github.com/eqcorrscan/EQcorrscan). 

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

Time series plotting  
```
fig = plot_Time_Series(doi, doi_end, files_path, filter=filter,
		       bandpass=bandpass, time_markers=time_markers)
```
![](doc/images/ts.png?raw=true)

### sigpig.core.lidar processes lidar data:  
3D point cloud plotting  
```
filename = "/Users/human/Dropbox/Programs/lidar/RR_2019_10cm_NAD83_UTMz10.laz"
v = visualize_Point_Cloud(filename)
```
![](doc/images/point_cloud.png?raw=true)

### sigpig.core.lfe_finder detections LFEs using matched-filtering:  
Process arrival time picks from a Snuffler marker file into detections
```
# define template length and prepick length (both in seconds)
template_length = 16.0
template_prepick = 0.5

# build stream of all station files for templates
files_path = "/Users/human/Dropbox/Research/Alaska/build_templates/2016-09-26"
doi = UTCDateTime("2016-09-26T00:00:00.0Z")
template_files = glob.glob(f"{files_path}/*.{doi.year}-{doi.month:02}"
			      f"-{doi.day:02}.ms")

# get templates and station_dict objects from picks in marker file
marker_file_path = "lfe_template.mrkr"
prepick_offset = 11 # in seconds
templates, station_dict, pick_offset = markers_to_template(marker_file_path, prepick_offset)

# define path of files for detection
detection_files_path = "/Volumes/DISK/alaska/data"
# define period of interest for detection
start_date = UTCDateTime("2016-06-15T00:00:00.0Z")
end_date = UTCDateTime("2018-08-11T23:59:59.9999999999999Z")

# run matched-filter detection
party = detect_LFEs(templates, template_files, station_dict,
                    template_length, template_prepick,
                    detection_files_path, start_date, end_date)
		    
# inspect the party object detections
detections_fig = party.plot(plot_grouped=True)
rate_fig = party.plot(plot_grouped=True, rate=True)
```
![](doc/images/lfe_detections.png?raw=true)

![](doc/images/lfe_detection_rate.png?raw=true)

Stack template detections
```
# load previous stream list?
load_stream_list = False
# get the stacks
stack_list = stack_waveforms(party, pick_offset,
		       	     detection_files_path, template_length,
			     template_prepick, station_dict)
```

### sigpig.core.autopicker detects and associates signals in time series:  
xyz  

### sigpig.core.stress inverts slip vectors for principal stress orientation:  
xy
