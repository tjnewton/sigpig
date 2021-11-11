%% Make an srStation structure for Rattlesnake

clear variables
clf
resolvepaths

addpath    ([work_dir,'srInput'],'-end');

%% Create srStation.  Not sure about Type yet? Type 2 source (borehole)

% load station file
% Format is station ID, lat, lon, utm x, utm y, elevation (m)
% load stalocs_1.dat
load ca_stalocs_1.dat
stalocs_1 = ca_stalocs_1;

srStation.name      = strtrim(cellstr( int2str(stalocs_1(:,1))));
srStation.easting   = stalocs_1(:,4)/1000;
srStation.northing  = stalocs_1(:,5)/1000;
srStation.elevation = stalocs_1(:,6)/1000;

a=unique(srStation.name);
if length(a) ~= length(srStation.name)
    disp('Unique naming problem')
    keyboard
end

save ../srInput/srStation_RS_CA srStation


