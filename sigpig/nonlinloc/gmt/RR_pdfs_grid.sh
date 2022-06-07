# get rid of degree,minute,second annotations
gmt set FORMAT_GEO_OUT=-D FORMAT_GEO_MAP=-D FORMAT_FLOAT_OUT=%.6g

# set the velocity model range of interest
# VEL_RANGE=0.3-0.45
# VEL_RANGE=0.4-0.55
# VEL_RANGE=0.5-0.65 # best based on number of accepted events
VEL_RANGE=0.6-0.75

# figure of Rattlesnake Ridge events + topography
gmt begin pdfs_$VEL_RANGE png

	# first subplot
	gmt subplot begin 2x2 -Fs15c,7c/25c,5c -A
	# generate a basemap for the region
	gmt basemap -Rk694.150000/694.448000/5155.400000/5155.898000 -JU+10/? -Baf -BWeNs+t"Event PDF Locations Vp=$VEL_RANGE"

	# # set lighting options for hillshade
	gmt grdgradient gridded_rr_dtm.nc -A310 -Nt0.6 -Grelief_gradient.nc # -A320

	# # make a custom colormap from the built-in dem2 colormap
	gmt makecpt -Cdem2 -T290/700

	# # # plot relief on basemap from built-in earth_relief data
	# # gmt grdimage @earth_relief_01s -I+d # 01s is too coarse for RR area

	# plot relief on basemap from downloaded DTM
	gmt grdimage gridded_rr_dtm.nc -C -Irelief_gradient.nc -t15 # -I+a320+nt0.6+m0 -Cdem4,

	# trim and plot the colorbar
	gmt colorbar -DJMR+o11.0c/-6+w10c/0.3c -B50 -Bx+l"Elevation (m)" -G290/520

	# get sums of pdf weights (-Ss), on 1 m grid in x and y (-I1e/1e)
	# gmt blockmean xyw_$VEL_RANGE.csv -I10e/10e -Ss >> counts_$VEL_RANGE.xyz

	# make a grid from the point weight counts
	# gmt surface counts_$VEL_RANGE.xyz -Ggridded_pdf_$VEL_RANGE.nc -T0.5 -I10e/10e
	# trim the interpolated grid
	# gmt grdfilter gridded_pdf_$VEL_RANGE.nc -Gtrim_gridded_pdf_$VEL_RANGE.nc -D0 -Fc1
	# make a custom colormap from a built-in colormap
	gmt makecpt -Cabyss -T0/100000  # -H >> pdfs.cpt
	# plot the grid
	gmt grdview gridded_rr_pdfs.nc -C -Qs -t40 #m
	# trim and plot the colorbar
	gmt colorbar -C -DJMR+o11.0c/6+w10c/0.3c -B10000 -Bx+l"PDF sum" # -G0/50000


	# plot stations at triangles
	gmt plot station.locs -Gred -Si0.4 -W0.1,black

	# build subplot with depth & longitude of locations
	gmt subplot set 1,0 -Cn-1.3
		# make a custom colormap from a built-in colormap
		gmt makecpt -Cabyss -T0/100000
		# plot the grid
		gmt grdview gridded_rr_pdfs_xz.nc -C -Qs -t40 -R-120.4683/-120.4646/-150/5 -JX? -Bx+l"Longitude" -By+l"Depth (m)" -BWrtS -W0.1,black # -Baf  
		# trim and plot the colorbar
		gmt colorbar -C -DJMR+o0.8c/0.1+w5c/0.3c -B10000 -Bx+l"PDF sum" # -G0/50000
	# gmt subplot end

	# build subplot with depth & latitude of locations
	gmt subplot set 0,1 -Cw-1.0
		# make a custom colormap from a built-in colormap
		gmt makecpt -Cabyss -T0/100000
		# plot the grid
		gmt grdview gridded_rr_pdfs_yz.nc -C -Qs -t40 -R-5/150/46.52415/46.52843 -JX7c/25c -Bx+l"Depth(m)" -By+l"Latitude" -BEwtS -W0.1,black # -Baf  
		# trim and plot the colorbar
		# gmt colorbar -C -DJMR+o0.8c/6+w10c/0.3c -B1000 -Bx+l"PDF sum" # -G0/50000
	gmt subplot end

gmt end show
