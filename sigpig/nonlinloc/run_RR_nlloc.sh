# echo "============"
# echo "Script to run sample locations for NonLinLoc - Non-Global mode"
# echo "see http://alomax.net/nlloc"
# echo "uses data from USGS and Alaska Earthquake Center"
# echo
# echo "IMPORTANT:  Requires:"
# echo "   1. NonLinLoc - the command \"NLLoc\" must be on your path"
# echo "   2. Java - the command \"java\" must be on your path"
# echo "   3. GMT4 - for postscript visualization, GMT4 must be installed and GMT4 tool commands must be on your path"
# echo "   4. SeismicityViewer must be installed and on your java classpath - see: http://alomax.net/seismicity"
# echo "   5. The environment variable PS_VIEWER is set in your shell or in this script"
# echo

# alias ghostview="open -a Preview"    # MacOSX
# PS_VIEWER=ghostview    # Linux, MacOSX
# setGMT4    # make sure GMT4 and not GMT5 commands are on path


# echo
# echo "Generate the model grid"
# Vel2Grid run/RR_nlloc.in
# echo

# echo "Visualize the model grid"
# Grid2GMT run/RR_nlloc.in model/RR.P.mod gmt/ V G 1 250 300 250
# gmt script needs to be edited before running
# ${PS_VIEWER} gmt/RR.P.mod.VG.ps &
# echo

# echo "Generate and view the travel-time and take-off angle grids "
# Grid2Time run/RR_nlloc.in
# echo
# echo "Visualize P travel-time grid"
# Grid2GMT run/RR_nlloc.in time/RR.P.32.time gmt/ V G 1 250 300 250
# ${PS_VIEWER} gmt/RR.P.32.time.VG.ps &
# echo

# echo "Visualize P take-off angles grid"
# Grid2GMT run/RR_nlloc.in time/RR.P.32.angle gmt/ V G 0 0 0 401
# ${PS_VIEWER} gmt/RR.P.32.angle.VG.ps &
# echo

# echo "Generate some synthetic arrival times "
# Time2EQ run/RR_nlloc.in
# more obs_synth/synth.obs
# echo

echo "Run NLL location "
NLLoc run/RR_nlloc.in
# echo "Run NLL-SSST-coherence travel time grid correction"
# Loc2ssst run/RR_nlloc.in
# echo "Run NLL relocation "
# NLLoc run/RR_nlloc_reloc.in
# combine locations
LocSum loc/RR.sum.grid0.loc 1 loc/RR "loc/RR.*.*.grid0.loc"

# echo "Plot the first event location with GMT"
# Grid2GMT run/RR_nlloc.in loc/RR.20180313.000125.grid0.loc gmt/ L S
# ${PS_VIEWER} gmt/alaska.20181130.172935.grid0.loc.LS.ps &
# echo

# echo "Plot the combined locations with GMT"
# LocSum loc/RR.sum.grid0.loc 1 loc/RR "loc/RR.*.*.grid0.loc"
# Grid2GMT run/RR_nlloc.in loc/RR gmt/ L E101
# ${PS_VIEWER} gmt/alaska.LE_101.ps &
# echo "Visualise the location with Seismicity Viewer (you must have installed Seismicity Viewer, see Seismicity Viewer software guide) "
# java net.alomax.seismicity.Seismicity loc/RR.*.*.grid0.loc.hyp

