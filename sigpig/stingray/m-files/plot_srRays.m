%  Plot some stuff from srRays in easting-northings

clear variables; close all; format short g

load ../srInput/srStation_RS_CA.mat
nsta = length(srStation.name);

for jj = 1:nsta
    file_in = ['../srOutput/srRays_',char(srStation.name(jj)),'.mat'];
    load(file_in)
    easting = srRays.srGeometry.easting + srRays.xg;
    northing= srRays.srGeometry.northing+ srRays.yg;
    [NN, EE] = meshgrid(northing, easting);
    contourf(EE,NN,srRays.time(:,:,1),24)
    hold on
    plot(srStation.easting(jj),srStation.northing(jj),'r*')
    hold off
    colorbar
    axis image
    pause
end
