%% Script for running TomoLab on Talapas (and locally with changes)

%% SETUP

clear ; close all; format short g; clc;
set(0,'DefaultFigureWindowStyle','docked')

% Get main directory
cd ../
work_dir = [pwd,'/'];
cd m-files;

% For Talapas
% rmpath(genpath(getenv('STINGRAY')));
setenv('STINGRAY','/gpfs/projects/seismolab/shared/Stingray')
setenv('STINGRAY_GIT_BPV','/gpfs/projects/seismolab/drt/StingrayGIT_BPV')

% Set environment variables and add paths
setenv('TAUPJAR',[getenv('STINGRAY_GIT_BPV'),'/contrib/TAUP/lib/TauP-2.4.5.jar']);
addpath([getenv('STINGRAY'),'/bin']);
addpath([getenv('STINGRAY_GIT_BPV'),'/toolbox']);
addpath([getenv('STINGRAY_GIT_BPV'),'/toolbox/utils']);
addpath(genpath([getenv('STINGRAY_GIT_BPV'),'/contrib']));

% Add input variable paths
addpath([work_dir,'/srInput/']);
addpath([work_dir,'/srOutput/']);

% Add/Remove revised codes
% rmpath(genpath([getenv('STINGRAY'),'/build']));
% addpath([getenv('STINGRAY'),'/build'],'-begin');

%% INPUT

res=1000

% Stingray/TomoLab Structures
theControl     = which('srControl_AMT.mat');
theGeometry    = which('srGeometry_AMT.mat');
theEvent       = which('srEvent_AMT.mat');
theStation     = which('srStation_AMT.mat');
theElevation   = which('srElevation_AMT.mat');
theModel       = which('srModel_1000_AMT.mat');

%% Load StingRay structures

srControl   = load_srControl(theControl);
srGeometry  = load_srGeometry(theGeometry);
srStation   = load_srStation(theStation,srGeometry);
srElevation = load_srElevation(theElevation,srGeometry);
srEvent     = load_srEvent(theEvent,srGeometry,srElevation);
srModel     = load_srModel(theModel,srControl,srGeometry,srElevation);
srArc       = arc_prep(srControl.arcfile, srModel.gx, srModel.gy, srModel.gz);

%%  Plots for checking setup
%{
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
plot(srElevation.LON(:),srElevation.LAT(:),'x');hold on
plot(srModel.LON(:),srModel.LAT(:),'r.')
plot(srGeometry.longitude,srGeometry.latitude,'*g')
plot([srStation.longitude],[srStation.latitude],'ok')
plot([srEvent.longitude],[srEvent.latitude],'.c')
legend({'Elevation nodes' 'Graph nodes' 'Graph origin' 'Station location'})
axis image

%  Plot fig_Model window (slowness and elevation)

figure(fig_Model)
clf
subplot(211)
pcolor(srModel.xg,srModel.yg,double(srModel.P.u(:,:,10)));
shading interp; axis image
colorbar
title('Slowness Model')

subplot(212)
pcolor(srModel.xg,srModel.yg,srModel.elevation(:,:,1));
shading interp; axis image
colorbar
title('Elevation model for graph')
drawnow

% pcolor(srModel.xg,srModel.yg,srModel.u(:,:,1)');
% xlabel('X-axis, km');ylabel('Y-axis, km');shading interp; axis image
%}
%% Intialize for an station (type 1)

for jj = 1 %:length(srStation.name)
    
    aStation = char(srStation.name(jj));
    xs = srStation.x(jj);
    ys = srStation.y(jj);
    
    % Init the srRays structure for a phase set defined by aPhase.
    % Currently assumes that it is P or S phase all along the path
    
    srRays = init_srRays('P');
    PorS   = char(srRays.model(1));
    
    [srInitialize]     = initFS(srModel,PorS,srArc,srControl,xs,ys,0); %,srStation.elevation(jj));
    
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
    
    fileout = [work_dir,'/srOutput/srRays_',char(srStation.name(jj)),'_',num2str(res),'.mat'];
    
    if ~exist(fileout,'file')
        
        % Direct phases; no carving at all!
        
        [srRays] = stingray(srModel,PorS,srInitialize,srArc,srControl,srRays,1);
        
        %  save the srRays structure
        
        % save(fileout,'srRays')
	save(fileout,'srRays','-V7.3')
    else
        disp(['File exists:  ',fileout])
    end

end

%%  Plot the result
%{
doplot = 0;

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
%}
