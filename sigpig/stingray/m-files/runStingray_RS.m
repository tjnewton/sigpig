%% Run Stingray for Rattlesnake

clear variables; close all; format short g
% resolvepaths
% cd ../
work_dir = [pwd,'/'];
cd m-files
addpath ('../srInput/','-end');
addpath ('../../stingray/toolbox/','-end');

%% Note on running for a slower velocity (10/11/20)
%  Running for a velocity of 600 m/s with 1 m/s/m increase in velocity.
%  Running locally on my laptop.  Had to make the following kluges for
%  quick run:
%    1) ON line 68 in ~/Dropbox/StinrayGIT/toolbox/stingray.m commented
%       out the line that uses srModel.anis_sym and just passed
%       srModel.P.anis_fraction(;), which is empty
%    2) copied over cp stingray_gateway.mexmaci64 ../../StingrayGIT/bin
%       from /Users/drt/Dropbox/StingrayGIT_LocalEQ/bin since I know this
%       version works on current OS.
%    3) After running, fixed the line in StingrayGIT/toolbox/stingray.m.
%       Also copied kluge version to RattleSnake/m-files 

% % Add Stingray paths
%
% addpath    ([work_dir,'/srInput'],'-end');

%% Input

theControl     = which('srControl_RS.mat');
theGeometry    = which('srGeometry_RS.mat');
theStation     = which('srStation_RS_CA.mat');
theModel       = which('srModel_RS_1D600.mat');
theElevation   = which('srElevation_RS.mat');
% % theArrival     = which('tlArrival_Hansa_subset_5p.mat');

%% Load StingRay structures

srControl   = load_srControl(theControl);
srGeometry  = load_srGeometry(theGeometry);
srStation   = load_srStation(theStation,srGeometry);
srElevation = load_srElevation(theElevation,srGeometry);
srModel     = load_srModel(theModel,srControl,srGeometry,srElevation);
srArc       = arc_prep(srControl.arcfile, srModel.gx, srModel.gy, srModel.gz);

% tlArrival   = load_tlArrival(theArrival);

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
plot(srElevation.EASTING(1:10:end),srElevation.NORTHING(1:10:end),'x');hold on
plot(srModel.EASTING(:),srModel.NORTHING(:),'r.')
plot(srGeometry.easting,srGeometry.northing,'*g')
plot([srStation.easting],[srStation.northing],'ok')
legend({'Elevation nodes' 'Graph nodes' 'Graph origin' 'Station location'})
axis image

%  Plot fig_Model window (slowness and elevation)

figure(fig_Model)
clf
subplot(211)
pcolor(srModel.EASTING,srModel.NORTHING,double(srModel.P.u(:,:,10)));
shading interp; axis image
colorbar
title('Slowness Model')

subplot(212)
pcolor(srModel.EASTING,srModel.NORTHING,srModel.elevation(:,:,1));
shading interp; axis image
colorbar
title('Elevation model for graph')
drawnow

% pcolor(srModel.xg,srModel.yg,srModel.u(:,:,1)');
% xlabel('X-axis, km');ylabel('Y-axis, km');shading interp; axis image
%}
%% Intialize for an station (type 1)

parfor jj = 1:length(srStation.name)
    
    aStation = char(srStation.name(jj));
    xs = srStation.x(jj);
    ys = srStation.y(jj);

    % Init the srRays structure for a phase set defined by aPhase.
    % Currently assumes that it is P or S phase all along the path
    
    srRays = init_srRays('Pg');
    PorS   = char(srRays.model(1));
    
    [srInitialize]  = initFS(srModel,PorS,srArc,srControl,xs,ys,0,srStation.elevation(jj));
    
%     %  Carve out some regions
%     
%     subArrival   = subset_tlArrival(tlArrival, 'P', [], srEvent.id(jj), 2);
%     [~,IA]       = intersect(srStation.name,subArrival.station);
%     xpts         = [srStation.x(IA);srEvent.x(jj)];
%     ypts         = [srStation.y(IA);srEvent.y(jj)];
%     if srControl.tf_carve
%         srInitialize = carveModel(srModel,srInitialize,xpts,ypts,...
%             srControl.carve.zvalue,srControl.carve.buffer);
%     end

    %% call stingray and fill srRays
    
    display(['Starting job:  ', int2str(jj)])
    
    fileout = [work_dir,'srOutput/','srRays_',char(srStation.name(jj)),'.mat'];
    
    if ~exist(fileout,'file')
        
        % Direct phases; no carving at all!
        
        [srRays] = stingray(srModel,srInitialize,srArc,srControl,srRays,1);
        
        %  save the srRays structure
        
        save_srRays(fileout,srRays)
    else
        disp(['File exists:  ',fileout])
    end

end

%%  Plot the result

doplot = 1;
srRays = load('/Users/human/Dropbox/Programs/location/srOutput/srRays_106.mat');

if doplot
    
    figure(fig_Time)
    subplot(211)
    pcolor(srRays.xg,srRays.zg,squeeze(srModel.u(:,15,:))')
    colormap(flipud(jet))
    hold on
    shading interp
    cc=caxis;
    cs=contour(srRays.xg,srRays.zg,squeeze(srRays.time(:,15,:))',[0:.1:11],'-w','LineWidth',2);
    clabel(cs)
    caxis(cc)
    axis image
    set(gca,'fontsize',18)
    xlabel('Kilometers','fontsize',18)
    ylabel('Kilometers','fontsize',18)
    
    subplot(212)
    pcolor(srRays.yg,srRays.zg,squeeze(srModel.u(57,:,:))')
    colormap(flipud(jet))
    hold on
    shading interp
    cc=caxis;
    cs=contour(srRays.yg,srRays.zg,squeeze(srRays.time(57,:,:))',[0:.1:11],'-w','LineWidth',2);
    clabel(cs)
    caxis(cc)
    axis image
    set(gca,'fontsize',18)
    xlabel('Kilometers','fontsize',18)
    ylabel('Kilometers','fontsize',18)
    figure(gcf)
    hold off
    
end

%%  Output a result (need a structure for the result of ray tracing.  Not
%   sure what this will be.  Need to record:
%   Velocity srModel
%   Station
%   times and iprec
%  Save Rays_name srRays
