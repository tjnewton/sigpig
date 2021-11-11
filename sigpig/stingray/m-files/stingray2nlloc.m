clear all;

% Written by Zoe Krauss

% Takes a Stingray travel time grid, which is an srRays variable saved as a matfile, and
% converts it to a travel time grid used by NLLoc

% .mat variable (srRays.time) --> binary .buf file

% Note: for travel time grid to work with Stingray, needs to be accompanied
% by a header .hdr file

%% SPECIFY INPUT AND OUTPUT FILES

% Stingray travel time grid:
load('srRays_singlestationS_NCHR.mat');

% Travel time grid for NLLOC:
% Format: (label).PHASE.STATIONCODE.time.buf
% Label can be anything you want
output_name = 'ENDsingle.S.NCHR.time.buf';

%% RESTRUCTURE TT GRID, SAVE AS NLLOC .buf FILE
fileid = fopen(output_name,'w');


[nx,ny,nz] = size(srRays.time);


index = 1;
for k = 1:nx
    for j = 1:ny
        for i = 1:nz
            node_time = srRays.time(k,j,i);
            fwrite(fileid,node_time,'float');
        end
    end
end


fclose(fileid)

