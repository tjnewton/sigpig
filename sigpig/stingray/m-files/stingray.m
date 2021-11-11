function [srRays] = stingray(srModel, srInitialize, srArc, srControl, srRays, N)

%STINGRAY - Calculates travel times and ray paths using graph theory.
%             (Stingray toolbox)
%
%   srRays = stingray(srModel,srInitialize,srArc,srControl) returns the
%   result in the first plane of a new structure srRays.
%
%   srRays = stingray(srModel,srInitialize,srArc,srControl,srRays,N)
%   returns the result in the N'th plane of the input structure srRays.
%
%   INPUT:
%
%           srModel:            Stingray structure
%           srInitialize:       Stingray structure
%           srArc:              Stingray structure
%           srControl:          Stingray structure
%
%   OPTIONAL INPUT:
%
%           srRays:             Stingray structure
%                               Stingray(N).model specifies P or S model.
%                               If 4 input arguments, N=1 and P is assumed.
%           N:                  N'th plane where results should be put.
%
%   OUTPUT:
%
%           srRays:             Stingray structure
%
%
%   Stingray is a matlab pre-processor for the stingray mexmaci file.
%   Fortran code is not double precision.  This function fills the single
%   precision variables, using data from the Stingray structures:
%
%   The %val construct is used inside stingray_gateway.
%   stingray_gateway operates directly on t, iprec and inSetS. It is not
%   recommended to call stingray_gateway from outside this m-file, in order
%   to avoid memory problems.
%
%  After calling stingray the srRays structure is filled out
%
%  Modified to account for anisotropy symmetry system (GMA, 2018).

%  Copyright 2010 Blue Tech Seismics, Inc.

%% Check number of input arguments

if nargin < 4, help(mfilename), error('Not enough arguments');end
if nargin ==5, help(mfilename),error('Specify number plane for srRays');end
if nargin ==4
    N = 1; 
    PorS = 'P';
else
    PorS = char(srRays(N).model);
end

%% Convert variables to fortan compatible declarations

%  Convert structure srModel to single precision variables.

u            = single(srModel.(PorS).u(:));
ghead        = single(srModel.ghead(:));

%%%%% Changed to account for anisotropy symmetry system (GMA, 2018).
%%%%% Negative represents slow symmetry system (i.e., cracks), positive
%%%%% represents fast symmetry system (i.e., olivine).

%######### first line is right; second temp for Rattlesnake
% a_r          = single(srModel.anis_sym(:) .* srModel.P.anis_fraction(:)); 
a_r          = single( srModel.P.anis_fraction(:)); 
%%%%%%%%%

a_t          = single(srModel.(PorS).anis_theta(:));
a_p          = single(srModel.(PorS).anis_phi(:));
knx          = int32(srModel.nx);
kny          = int32(srModel.ny);
knodes       = int32(srModel.nodes);

%  Convert structure srInitialize to single precision

inSetS       = int32(srInitialize.inSetS);
t            = single(srInitialize.time);
iprec        = int32(srInitialize.iprec);

%  Convert topography to single precision

zhang        = single(srModel.elevation);

%  Convert flags to single precision

tf_anisotropy     = int32(srControl.tf_anisotropy);
tf_line_integrate = int32(srControl.tf_line_integrate);

%  Convert srArc structure to single precision.

arcList      = int32(srArc.arcList);
arcHead      = int32([srArc.mx srArc.my srArc.mz srArc.nfs]);
kmx          = int32(srArc.mx);
kmy          = int32(srArc.my);
kmz          = int32(srArc.mz);
kmaxfs       = int32(srArc.nfs);

%%  Call mex file for ray tracing.

% stingray_gateway(u,ghead,inSetS,arcList,arcHead,...
%     zhang,tf_arcweight,t,iprec,...
%     knx,kny,kmx,kmy,kmz,kmaxfs,knodes,...
%     a_r,a_t,a_p);   %passing 19 arguments

tic
stingray_gateway(u, t, iprec, inSetS, ...
    ghead, knx, kny, knodes,...
    arcList, arcHead, kmx, kmy, kmz, kmaxfs, ...
    zhang, tf_line_integrate, ...
    tf_anisotropy, a_r, a_t, a_p);   %passing 20 arguments
toc

%%  Fill srRays structure
%
%  Required fields:
%
%       srRays.ghead               (1:8)
%       srRays.time                (nx, ny, nz)
%       srRays.iprec               (nx, ny, nz)
%
%  Derived fields:
%
%       srRays.nx                  nodes in x-direction
%       srRays.ny                  nodes in y-direction
%       srRays.nz                  nodes in z-direction
%       srRays.gx                  node-spacing in x
%       srRays.gy                  node-spacing in y
%       srRays.gz                  node-spacing in z
%       srRays.nodes               total number of nodes
%       srRays.xg                  x-location of nodes
%       srRays.yg                  y-location of nodes
%       srRays.zg                  z-location of nodes
%       srRays.elevation           mesh of elevation at nodes
%       srRays.srGeometry          srGeometry holds origin and rotaiton
%       srRays.modelname           srModel.filename (velocity model)
%
%  srRays should be able to describe itself completely.

%%  fill srRays

srRays(N).ghead      = srModel.ghead;
srRays(N).time       = double(reshape(t,srModel.nx,srModel.ny,srModel.nz));
srRays(N).iprec      = double(reshape(iprec,srModel.nx,srModel.ny,srModel.nz));

srRays(N).nx         = srModel.nx;
srRays(N).ny         = srModel.ny;
srRays(N).nz         = srModel.nz;
srRays(N).gx         = srModel.gx;
srRays(N).gy         = srModel.gy;
srRays(N).gz         = srModel.gz;
srRays(N).nodes      = srModel.nodes;
srRays(N).xg         = srModel.xg;
srRays(N).yg         = srModel.yg;
srRays(N).zg         = srModel.zg;
srRays(N).elevation  = srModel.elevation;
srRays(N).srGeometry = srModel.srGeometry;
srRays(N).modelname  = srModel.filename;
srRays(N).ghead      = srModel.ghead;
srRays(N).srControl  = srControl;
