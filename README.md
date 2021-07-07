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

# bandpass filter from 2-8 Hz
bandpass = [2, 8]

fig = plot_Time_Series_And_Spectrogram(doi, doi_end, files_path, filter=True, bandpass=bandpass)
```
![](doc/images/ts-spect.png?raw=true)
