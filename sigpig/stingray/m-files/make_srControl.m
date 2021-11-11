%  Script to make required variables for srControl structure used by
%  Stingray.  Values are for VSP Lunding experiment
%

clear all

%% Enter values below

srControl.tf_latlon         = false;
srControl.tf_anisotropy     = false;
srControl.tf_line_integrate = false;
srControl.arcfile           = 'arc7.mat';
srControl.tf_waterpath      = 0;
srControl.waterpath.vel     = 1.480;
srControl.waterpath.xy_inc  = .050;
srControl.tf_carve          = 1;
srControl.carve.buffer      = .20;
srControl.carve.zvalue      = [];

%%

display(srControl)

reply = input('Save a file? y/n: ','s');
disp(' ')

if reply == 'y'
    filename = input('Enter a filename:  ','s');disp(' ')

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
        s = ['save ',filename,' srControl'];
        disp(' ')
        disp(['  Saving srControl to ',filename,'.mat']);
        eval(s);disp(' ')
    end
    
end