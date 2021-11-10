clear all;

% load station locations
load stalocs_1.dat

% loop over each station's srRays file
for n =[10 12 13 14 15 16 17 18 2 20 21 22 23 25 26 27 28 3 30 31 32 33 34 35 36 37 38 39 4 40 41 42 5 6 7 8 9 103 105 106]
    % missing 1, and 14 is additional, check iris, what does this mean for srRays files? how to generate?

    % define path to stingray travel time grid
    file=['/Users/amt/Documents/rattlesnake_ridge/ray_tracing/srRays_' num2str(n) '.mat']
    load(file);

    % travel time grid format for NLLOC: label.PHASE.STATIONCODE.time.buf
    % where label can be any characters
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

    ind=find(stalocs_1(:,1)==n)

    % build .hdr file
    file_id = fopen(['RR.P.' num2str(n) '.time.hdr'],'w');
    fprintf(file_id,'%d %d %d %f %f %f %f %f %f %s\n', srRays.nx, srRays.ny, srRays.nz, ...
        srRays.srGeometry.easting, srRays.srGeometry.northing, 0, srRays.gx, srRays.gy, srRays.gz, 'TIME');
    fprintf(file_id,'%s %f %f %f\n', num2str(n), stalocs_1(ind,4)/1000, stalocs_1(ind,5)/1000, 0); %-1*stalocs_1(ind,6)/1000);
    fclose(file_id);
end