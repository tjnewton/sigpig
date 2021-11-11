%  Resolve host for directory paths
%  Sets the variables data_dir and work_dir

host = getenv('HOST');

if strcmp(host,'gilchrist.uoregon.edu') 
    data_dir = '/Users/drt/Dropbox/Rattlesnake/';
    work_dir = '/Users/drt/Dropbox/Rattlesnake/';
    disp(['Running on :  ',host])
elseif strcmp(host(1:5),'Elvis')
    data_dir = '/Users/drt/Dropbox/Rattlesnake/';
    work_dir = '/Users/drt/Dropbox/Rattlesnake/';
    disp(['Running on :  ',host])
else
    disp('  Failed to resolve host');
    disp(' '); drawnow
    return
end

disp(' ')
