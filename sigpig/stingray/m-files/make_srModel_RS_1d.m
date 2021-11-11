%  Build srModel for RS
%  
%
%%
clear variables
clf
% resolvepaths
cd ../
work_dir = [pwd,'/'];
cd m-files
addpath ('../srInput/','-end');

% addpath    ([work_dir,'srInput'],'-end');

%% Limits of model.  Will grid at 2 m

minx = 694.15;
maxx = 694.45;
miny = 5155.40;
maxy = 5155.90;
maxz = 0;
minz = -0.150;
dxyz = 0.002;

srModel.ghead(1) = 0;
srModel.ghead(2) = 0;
srModel.ghead(3) = round((maxx-minx)/dxyz) +1;
srModel.ghead(4) = round((maxy-miny)/dxyz) +1;
srModel.ghead(5) = round((maxz-minz)/dxyz) +1;
srModel.ghead(6) = dxyz;
srModel.ghead(7) = dxyz;
srModel.ghead(8) = dxyz;

%%  Table for simple 1d model (depth; velocity)

v1d = [0,       .600; ...
       0.05,    .650; ...
       0.10,    .700; ...
       0.15,    .750; ];
      
%%  

theControl     = which('srControl_RS.mat');
theGeometry    = which('srGeometry_RS.mat');
theStation     = which('srStation_RS_CA.mat');
theElevation   = which('srElevation_RS.mat');

%% Load StingRay structures

srControl   = load_srControl(theControl);
srGeometry  = load_srGeometry(theGeometry);
srStation   = load_srStation(theStation,srGeometry);
srElevation = load_srElevation(theElevation,srGeometry);

%%  Plots for checking setup
% {
% Setup figures
set(0,'DefaultFigureWindowStyle','docked')
if exist('fig_Map','var')
    clf(fig_Map);
    clf(fig_Model);
    clf(fig_Time);
else
    fig_Map   = figure('Name','Map Window','NumberTitle','off');
    fig_Model = figure('Name','Model Window','NumberTitle','off');
    fig_Time  = figure('Name','Time Window','NumberTitle','off');
end

%  Plot fig_Map window

figure(fig_Map)
clf(fig_Map)
contourf(srElevation.EASTING,srElevation.NORTHING,srElevation.data);hold on
% plot(srModel.EASTING(:),srModel.NORTHING(:),'r.')
plot(srGeometry.easting,srGeometry.northing,'*g')
plot([srStation.easting],[srStation.northing],'ok')
load slidepoly.dat
plot(slidepoly(:,1)/1000,slidepoly(:,2)/1000,'ko','MarkerFaceColor','r','MarkerSize',10)


% plot([srEvent.easting],[srEvent.northing],'.c')
legend({'Elevation nodes' 'Graph nodes' 'Graph origin' 'Station location'})
axis image
% UTMlabels

%%


% Interpolate simple 1D model and make a 3D model out of it

z     = 0:srModel.ghead(8):abs(minz);
vv    = interp1(v1d(:,1),v1d(:,2),z);
vel3d = zeros(srModel.ghead(3),srModel.ghead(4),srModel.ghead(5));

for i = 1:srModel.ghead(3)
    for j = 1:srModel.ghead(4)
        vel3d(i,j,:) = vv';
    end
end

% srModel.P.u = (1/2.0)*ones(srModel.ghead(3),srModel.ghead(4),srModel.ghead(5));
srModel.P.u = 1./vel3d;

% save srModel_RS_1D srModel

