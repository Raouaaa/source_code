% Start/go to in the Data_77subjects folder containing the LH and RH folders
% -------------------------------------------------------------------------
function extractD77pview
% Extract all 4 views from file and add to a tensor and save it

cd 'Data_77subjects';
labels = importdata('labels77.mat');
cd 'LH77subjects';
LHmat1 = extractData(1); % extract view 1
LHmat2 = extractData(2);
LHmat3 = extractData(3);
LHmat4 = extractData(4);
cd ..;
cd 'RH77subjects';
RHmat1 = extractData(1);
RHmat2 = extractData(2);
RHmat3 = extractData(3);
RHmat4 = extractData(4);

cd '..\..';
save('LHmat1','LHmat1');
save('LHmat2','LHmat2');
save('LHmat3','LHmat3');
save('LHmat4','LHmat4');

save('RHmat1','RHmat1');
save('RHmat2','RHmat2');
save('RHmat3','RHmat3');
save('RHmat4','RHmat4');
end

% Paramter : view that will be saved
function mat = extractData(v)
fileInfo = dir('*.mat');
mat = zeros(77,595);
% For every file in current directory, load data from txt file
for file = 1 : length(fileInfo)
    filename = fileInfo(file).name;
    data = load(filename);
	
	% Construct tensor
    views = constructTensor(data,v); 
    temp = vertVectorise(views);
    mat(file,1:end)=temp;
  
end
end

% Construct 1-view tensor for the subject's passed in data
function views = constructTensor(data,v)
    views = zeros(35,35);
    views(:,:) = data.A(:,:,v);
end


% Take a view and turn it into a horizontal vector of the upper triangular
% half of the matrix, diagnal of zeros not included. 
function featVector = vertVectorise(views)

featVector = views(triu(true(size(views)),1));
featVector = reshape(featVector, [1, length(featVector)]);
end

