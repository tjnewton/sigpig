%  Reformatting the Elevation file to be Stingray compatible
%


%% Control

clear variables
resolvepaths
format short g
figure(1); clf

% Parameters

threshold = 10;


% load grid file
% easting, northing, z, (m)

[X,Y,Z]=grdread2('rr_utm.grd');
Z=double(Z);
[xgrd,ygrd]=meshgrid(X,Y);

%% Limits of region

minx = 694.0;
maxx = 694.73;
miny = 5155.2;
maxy = 5156.1;

%%  Make header; note dimensions in kilometers for header

srElevation.header(1) = minx;
srElevation.header(2) = maxx;
srElevation.header(3) = miny;
srElevation.header(4) = maxy;
srElevation.header(5) = 1/1000;
srElevation.header(6) = 1/1000;
srElevation.header(7) = (maxx-minx)*1000+1;
srElevation.header(8) = (maxy-miny)*1000+1;

%% Interpolate to 1 m grid

xint = srElevation.header(1)*1000:1:srElevation.header(2)*1000;
yint = srElevation.header(3)*1000:1:srElevation.header(4)*1000;

[YY,XX]=meshgrid(yint,xint);
srElevation.data = interp2(xgrd,ygrd,Z,XX,YY);

[n,m]=size(srElevation.data);

for i = 1:n
    a=diff(srElevation.data(i,:));
    plot(a,'ro')
    hold on
    axis([0 731 -100 100])
    I=find(abs(a)> threshold);
%     p=length(I);
%     if p~=0
%         for j=1:p
%             srElevation.data(i,I(j)) = (srElevation.data(i,I(j)-1)+srElevation.data(i,I(j)+1))/2;
%         end
%     end
%     b=diff(srElevation.data(i,:));
    srElevation.data(i,I) = nan;
    b=diff(srElevation.data(i,:));
    plot(b,'bo')
    hold off
    
    
end

%%  There are nans in the map; remove them by interpolation

x = XX(:);
y = YY(:);
z = srElevation.data(:);
J=~isnan(z);
x=x(J);
y=y(J);
z=z(J);
srElevation
srElevation.data = griddata(x,y,z,XX,YY,'linear');
srElevation
contourf(XX,YY,srElevation.data)

save srElevation_RS srElevation


