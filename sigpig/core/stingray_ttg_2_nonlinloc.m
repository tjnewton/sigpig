clear all;

% load station locations
load /Users/human/Dropbox/Research/Rattlesnake_Ridge/stingray_rr/station_locations.dat

% loop over each station's srRays file
for n =[10 12 13 1 15 16 17 18 2 20 21 22 23 25 26 27 28 3 30 31 32 33 34 35 36 37 38 39 4 40 41 42 5 6 7 8 9 103 105 106]

    % define path to stingray travel time grid
    file=['/Users/human/Dropbox/Research/Rattlesnake_Ridge/stingray_rr/srOutput/0.6-0.75/srRays_' num2str(n) '_2.mat']
    load(file);

    % travel time grid format for NLLOC: label.PHASE.STATIONCODE.time.buf, where label can be any characters
    output_name = ['RR.P.' num2str(n) '.time.buf'];

    % transform the travel time grid, save as nlloc .buf file
    file_id = fopen(output_name, 'w');
    [nx,ny,nz] = size(srRays.time);
    index = 1;
    for k = 1:nx
        for j = 1:ny
            for i = 1:nz
                node_time = srRays.time(k,j,i);
                fwrite(file_id,node_time,'float');
            end
        end
    end
    fclose(file_id)

    ind=find(station_locations(:,1)==n)

    % build .hdr file
    file_id = fopen(['RR.P.' num2str(n) '.time.hdr'],'w');
    fprintf(file_id,'%d %d %d %f %f %f %f %f %f %s\n', srRays.nx, srRays.ny, srRays.nz, ...
        srRays.srGeometry.easting, srRays.srGeometry.northing, 0, srRays.gx, srRays.gy, srRays.gz, 'TIME');
    fprintf(file_id,'%s %f %f %f\n', num2str(n), station_locations(ind,2), station_locations(ind,3), -1*station_locations(ind,4)); % or 0);
    fclose(file_id);
end