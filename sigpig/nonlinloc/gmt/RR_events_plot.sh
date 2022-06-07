# get rid of degree,minute,second annotations
gmt set FORMAT_GEO_MAP=D FORMAT_FLOAT_OUT=%.6lg

# set the velocity model range of interest
# VEL_RANGE=0.3-0.45
# VEL_RANGE=0.4-0.55
# VEL_RANGE=0.5-0.65 # best based on number of accepted events
VEL_RANGE=0.6-0.75

# get the number of events from one of the csv files, xargs strips whitespace
NUM_EVENTS=$(wc -l < x_z_$VEL_RANGE.csv | xargs)

# figure of Rattlesnake Ridge events + topography
gmt begin events_$VEL_RANGE png

	# first subplot
	gmt subplot begin 2x1 -Fs15c/25c,5c -A
	# generate a basemap for the region
	gmt basemap -Rk694.150000/694.448000/5155.400000/5155.898000 -JU+10/? -Baf -BWeNs+t"RMS=5.84 n=$NUM_EVENTS Vp=$VEL_RANGE"

	# set lighting options for hillshade
	gmt grdgradient gridded_rr_dtm.nc -A310 -Nt0.6 -Grelief_gradient.nc # -A320

	# make a custom colormap from the built-in dem2 colormap
	gmt makecpt -Cdem2 -T290/700

	# # plot relief on basemap from built-in earth_relief data
	# gmt grdimage @earth_relief_01s -I+d # 01s is too coarse for RR area

	# plot relief on basemap from downloaded DTM
	gmt grdimage gridded_rr_dtm.nc -C -Irelief_gradient.nc -t15 # -I+a320+nt0.6+m0 -Cdem4,

	# trim and plot the colorbar
	gmt colorbar -DJMR+o0.8c/0+w10c/0.3c -B50 -Bx+l"Elevation (m)" -G290/520

	# plot events as stars
	gmt plot x_y_horizUncert_$VEL_RANGE.csv -Gblack -Sa0.4 -W0.1,black

	# plot horizontal uncertainties
	gmt plot x_y_horizUncert_$VEL_RANGE.csv -Sc -W0.1,white -t70

	# build subplot with depth & longitude of locations
	gmt subplot set 1,0 -Cn-1.3
	gmt plot x_z_$VEL_RANGE.csv -Gblack -Sa0.4 -R-120.4683/-120.4646/-150/10 -JX? -Bx+l"Longitude" -By+l"Depth (m)" -BWrtS -W0.1,black # -Baf  

	gmt subplot end
gmt end show
