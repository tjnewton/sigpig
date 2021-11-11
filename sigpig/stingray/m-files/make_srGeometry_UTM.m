%  Script to make required variables for srGeometry structure used by
%  Stingray. 
%
%  This script is useful when the srGeometry.tf_latlon is false.  In
%  this case Stingray is using input files that have km-km values. 
%
%
%  srGeometry.longitude:  decimal degrees
%  srGeometry.latitude:   decimal degrees
%  srGeometry.rotation:   degrees
%
%  srGeometry.easting:    decimal degrees
%  srGeometry.northing:   decimal degrees
%  srGeometry.rotation:   degrees

%  Rattlesnake
%  minx, max

minx = 694.15;
maxx = 694.45;
miny = 5155.40;
maxy = 5155.90;

% Enter values below (UTM values for LUNO, but in km
srGeometry.tf_latlon = false;
srGeometry.easting   = minx;
srGeometry.northing  = miny;
srGeometry.rotation  = 0;

display(srGeometry)

reply = input('Do you want to save a file? y/n: ','s');
disp(' ')

if reply == 'y'
    filename = input('Enter a filename:  ','s');disp(' ');
    
    filename = deblank(filename);
    if strcmp(filename(end-3:end),'.mat')
        filename=filename(1:end-4);
    end
    
    s = ['exist(''',filename,'.mat''',')'];
    flag = eval(s);
    
    if flag == 2
        reply = input('File exists:  overwrite? y/n: ','s');
    end
    
    if flag==0 || reply=='y'
        s = ['save ',filename,' srGeometry'];
        disp(' ')
        disp(['  Saving srGeometry to ',filename,'.mat']);
        eval(s);disp(' ')
    end
    
end