clear variables; close all; format short g
resolvepaths

% % Add Stingray paths
%
addpath    ([work_dir,'/srInput'],'-end');

%% Input

theControl     = which('srControl_RS.mat');
theGeometry    = which('srGeometry_RS.mat');
theStation     = which('srStation_RS_CA.mat');
theModel       = which('srModel_RS_1D.mat');
theElevation   = which('srElevation_RS.mat');
% % theArrival     = which('tlArrival_Hansa_subset_5p.mat');

%% Load StingRay structures

srControl   = load_srControl(theControl);
srGeometry  = load_srGeometry(theGeometry);
srStation   = load_srStation(theStation,srGeometry);
srElevation = load_srElevation(theElevation,srGeometry);
srModel     = load_srModel(theModel,srControl,srGeometry,srElevation);
srArc       = arc_prep(srControl.arcfile, srModel.gx, srModel.gy, srModel.gz);

%%
dz = zeros(srStation.nsta,1);
for jj = 1:srStation.nsta
    aStation = char(srStation.name(jj));
    xs = srStation.x(jj);
    ys = srStation.y(jj);
    
    [iw,jw,kw] = xyz2ijk(xs, ys, 0, srModel.ghead);
    dz(jj) = srStation.elevation(jj)-srModel.elevation(iw,jw);
end

%%

dz = dz*1000;
mean(dz)
std(dz)

