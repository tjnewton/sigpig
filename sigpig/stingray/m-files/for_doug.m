close all
clearvars
format short g

%% Limits of region

minx = 694.0;
maxx = 694.73;
miny = 5155.2;
maxy = 5156.1;

%%

% load grid file
% easting, northing, z, (m)
[X,Y,Z]=grdread2('rr_utm.grd');
Z=double(Z);
[xgrd,ygrd]=meshgrid(X,Y);

% plot grid file
% figure
decfac=10;
surf(xgrd(1:decfac:end,1:decfac:end),ygrd(1:decfac:end,1:decfac:end),Z(1:decfac:end,1:decfac:end))
view(0,90)
shading interp
axis(1000* [ minx maxx miny maxy])
axis tight equal
hold on

% load station file
% Format is station ID, lat, lon, utm x, utm y, elevation (m)
load stalocs_1.dat

% plot stations
plot3(stalocs_1(:,4),stalocs_1(:,5),stalocs_1(:,6)+1,'k^','MarkerFaceColor','k','MarkerSize',10)

% draw polygon
% [xp,yp]=ginput
load slidepoly.dat
plot3(slidepoly(:,1),slidepoly(:,2),ones(1,length(slidepoly(:,2)))*1550,'ko','MarkerFaceColor','r','MarkerSize',10)

% source grid
minel=1035
xgrdvec=reshape(xgrd,1,numel(xgrd));
ygrdvec=reshape(ygrd,1,numel(ygrd));
zgrdvec=reshape(Z,1,numel(Z));

in = inpolygon(xgrdvec,ygrdvec,slidepoly(:,1),slidepoly(:,2));
source_x=xgrdvec(find(in));
source_y=ygrdvec(find(in));
source_z=zgrdvec(find(in));
pause
source_x=source_x(1:decfac:end);
source_y=source_y(1:decfac:end);
source_z=source_z(1:decfac:end);

%plot3(source_x,source_y,source_z,'ko','MarkerFaceColor','k','MarkerSize',4)
minel=1035;
source_vol=[]
for ii=1:length(source_x)
    ii
    if source_z(ii) < 1570 % kluge
        if minel > (source_z(ii)-100)
            elcol=minel:10:source_z(ii);
        else
            elcol=(source_z(ii)-100):10:source_z(ii);
        end
    end
    tmp=horzcat(source_x(ii)*ones(numel(elcol),1),source_y(ii)*ones(numel(elcol),1),elcol');
    source_vol=vertcat(source_vol,tmp);
end
figure
plot3(source_vol(:,1), source_vol(:,2), source_vol(:,3),'ko','MarkerFaceColor','k','MarkerSize',4)
axis tight equal
view(0,90)