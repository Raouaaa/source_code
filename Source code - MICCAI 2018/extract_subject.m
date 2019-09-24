function avg_mat = extract_subject(data)

cd 'Data_77subjects';
file_dest = strcat(data(1:2),'77subjects');
cd(file_dest)
view = str2num(data(6));
avg_mat = extractData(view);
cd '..\..';
end

function mat = extractData(v)
fileInfo = dir('*.mat');
mat = zeros(35,35);
% For every file in current directory, load data from txt file
    for file = 1 : length(fileInfo)
            filename = fileInfo(file).name;
            data = load(filename);
            mat = mat + data.A(:,:,v);
    end
    mat = mat/length(fileInfo);
end
