# sigpig
## Seismology and Signal Processing Toolkit 
This repository contains data pipelines developed for research. These pipelines generally start by ingesting data and/or models, followed by an analysis to infer some property of the system. Results are plotted via the sigpig.figures module, which contains numerous functions to visualize and describe data, models, and analyses. 
Sigpig is built on top of [matplotlib](https://github.com/matplotlib/matplotlib), [ObsPy](https://github.com/obspy/obspy), [GDAL](https://github.com/OSGeo/gdal), [Laspy](https://github.com/laspy/laspy), [NumPy](https://github.com/numpy/numpy), [pptk](https://github.com/heremaps/pptk), [EQcorrscan](https://github.com/eqcorrscan/EQcorrscan), and other libraries. 

### To get started:
Clone this repo in your preferred directory:  
`git clone https://github.com/tjnewton/sigpig.git`  
Move into the sigpig directory:  
`cd sigpig`  
Use conda to create a Python environment from the sigpig.yml file:  
`conda env create -f sigpig.yml`  
Activate the environment:  
`conda activate sigpig`  
Launch Jupyter Lab within sigpig environment:  
`jupyter lab`  
Browse the repo directories and notebooks within Jupyter Lab :)

## Functionality:
### sigpig.core.data fetches, formats, and analyzes data:  
Download time series data from IRIS DMC  
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

Fetch the geophone station locations for the Rattlesnake Ridge experiment on a specified date  
```
# specify the date of interest
date = UTCDateTime("2018-03-13T00:04:00.0Z")
# get station locations coordinates in UTM meters easting and northing format
format = "UTM"
station_locations = rattlesnake_Ridge_Station_Locations(date, format=format)
```

### sigpig.core.figures generates various figures:  
Figures are either generated from objects, like NumPy arrays and ObsPy streams, or from files via a list of file paths. Functions: plot_stalta, plot_stream, plot_Time_Series_And_Spectrogram, plot_Time_Series, plot_party_detections, plot_stack, plot_template_and_stack, plot_stream_absolute, plot_distribution, plot_event_picks, plot_trace_curvature, spectrogram, spectra.  

Time series and spectrogram plotting from files:  
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

Time series plotting from files:  
```
fig = plot_Time_Series(doi, doi_end, files_path, filter=filter,
		       bandpass=bandpass, time_markers=time_markers)
```
![](doc/images/ts.png?raw=true)

Spectrogram plotting from an Obspy stream object:  
```
fig = spectrogram(stream)
```

Spectra plotting from an Obspy stream object: 
```
fig = spectra(stream)
```

### sigpig.core.lidar processes lidar data:  
3D point cloud plotting  
```
filename = "/Users/human/Dropbox/Programs/lidar/RR_2019_10cm_NAD83_UTMz10.laz"
v = visualize_Point_Cloud(filename)
```
![](doc/images/point_cloud.png?raw=true)

Retrieve elevations from raster at specified points    
```
raster_file = '/Users/human/Dropbox/Programs/lidar/yakima_basin_2018_dtm_43.tif'
longitudes = [-120.480, -120.480]
latitudes = [46.538, 46.519]

# query raster at specified coordinates
elevations = elevations_from_raster(raster_file, longitudes, latitudes)
```

Query raster on a grid    
```
raster_file = '/Users/human/Dropbox/Programs/lidar/yakima_basin_2018_dtm_43.tif'
# define limits of grid
x_limits = [694.15, 694.45]
y_limits = [5155.40, 5155.90]

# query raster on a grid
longitude_grid, latitude_grid, elevation_grid = grids_from_raster(raster_file, x_limits, 
				                                  y_limits, plot=False,
                                                                  UTM=True)
```

### sigpig.core.lfe_finder detects LFEs using matched-filtering:  
Process arrival time picks from a Snuffler marker file into detections
#FIXME: add single call function for detection
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

### sigpig.core.autopicker detects noise windows in time series:  
#TODO: rename this to noise_miner, as autopicker is refactored into time_miner and noise_miner

### sigpig.core.time_miner detects and associates signals in time series:  
#TODO: rename to something more intuitive like event_miner and refactor internal names

Detect phase arrival times and associate phases to events:  
```
# run time_miner.py from terminal and specify the start and end dates of the detection as parameters
# this example specifies a 5 second window
./time_miner.py 2018-03-13T00:10:00.0Z 2018-03-13T00:10:05.0Z

# set parameters in time_miner.py for tunability, including plot_Association_Results=True to generate figures
```
![](doc/images/association_results.png?raw=true)  

Process an autopicked event catalog and display the results:  
```
from data import top_n_autopicked_events
import pickle

# define the path to the autopicked events file in snuffler format
autopicked_file_path = "autopicked_events_03_13-07_09_2018.mrkr"
# define the desired number of events to get (-1 means all events)
n = -1
# extract a dictionary of events processed from the file
events = top_n_autopicked_events(autopicked_file_path, n)

# save events dict to a file for future reference
events_dict_path = "events_dict.pkl"
outfile = open(events_dict_path, 'wb')
pickle.dump(events, outfile)
outfile.close()

# generate a histogram to describe the events
events_dict_histogram(events_dict_path, save_fig=True)
```
![](doc/images/events_histogram.png?raw=true)  

### sigpig.core.stress inverts slip vectors for principal stress orientation:  
xy
