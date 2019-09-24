% Run the SVM test for each FS method
% Run it 10 times for k=10 to k=100 with a step of 10
% Return a table of the average result for each k and each FS method

function sTable = create_tab(data)

	clc;
    %Remove warnings
	warning('off', 'stats:obsolete:ReplaceThisWithMethodOfObjectReturnedBy');
	warning('off', 'stats:obsolete:ReplaceThisWith');
    warning('off', 'stats:svmclassify:NoTrainingFigure');
    warning('off', 'stats:svmtrain:OnlyPlot2D');
    warning('off', 'MATLAB:singularMatrix');
    warning('off', 'MATLAB:illConditionedMatrix');
    warning('off', 'MATLAB:nearlySingularMatrix');
	% Include dependencies
	addpath('./lib'); % dependencies
	addpath('./methods'); % FS methods
	addpath(genpath('./lib/drtoolbox'));

	% Load the data and select features for classification

	tab_res=zeros(7,10);
    temp_res=zeros(7,10);
	X = importdata(data);
    	
	% Extract labels
	labels = importdata('labels.mat');
	Y = labels;

	
	% Randomly partitions observations into a training set and a test
	% set using stratified holdout
	
    %create a loop 
    %for fold = 1:size(X,1)
    c = cvpartition(Y,'KFold',10);
    for kfold = 1:10
        
        new_Y = Y(c.training(kfold));
        new_X = X(c.training(kfold),:);
        for fold = 1:c.TrainSize(kfold)
            P = cvpartition(new_Y,'LeaveOut');
            X_train = new_X(training(P,fold),:);
            Y_train = new_Y(training(P,fold),:);

            X_test = new_X(P.test(fold),:);
            Y_test = new_Y(P.test(fold),:);

            % number of features
            numF = size(X,2); % number of features, here 595

            %relieff
            %nested cross validation (20)
            max_rate=0;
            best_k=5;
            P_NCV = cvpartition(Y_train,'KFold',5);
            for k=10:10:50
                acc_final=0;
                for fold_ncv = 1:5
                    X_train_ncv = new_X(training(P_NCV,fold_ncv),:);
                    Y_train_ncv = new_Y(training(P_NCV,fold_ncv),:);
                    X_test_ncv = new_X(P_NCV.test(fold_ncv),:);
                    Y_test_ncv = new_Y(P_NCV.test(fold_ncv),:);
                    [ranking, w] = reliefF( X_train_ncv, Y_train_ncv, k);
                    acc_tmp = ncv(X_train_ncv,Y_train_ncv,X_test_ncv, Y_test_ncv,numF,P_NCV,ranking);
                    acc_final = acc_final + acc_tmp;
                end 
                acc_final= acc_final / 5;
                if acc_final > max_rate
                    max_rate=acc_final;
                    best_k = k;
                end
            end

            [ranking, w] = reliefF( X_train, Y_train, best_k);
            temp = get_rate(X_train,Y_train,X_test, Y_test,numF,P,ranking);
            tab_res(1,:) = tab_res(1,:) + temp;

            %mutinff
            [ ranking , w] = mutInfFS( X_train, Y_train, numF );
            temp = get_rate(X_train,Y_train,X_test, Y_test,numF,P,ranking);
            tab_res(2,:) = tab_res(2,:) + temp;

            %fsv
            %[ ranking , w] = fsvFS( X_train, Y_train, numF );
            %temp = get_rate(X_train,Y_train,X_test, Y_test,numF,P,ranking);
            %tab_res(3,:) = tab_res(3,:) + temp;

            %laplacian
            W = dist(X_train');
            W = -W./max(max(W)); % it's a similarity
            [lscores] = LaplacianScore(X_train, W);
            [junk, ranking] = sort(-lscores);
            temp = get_rate(X_train,Y_train,X_test, Y_test,numF,P,ranking);
            tab_res(3,:) = tab_res(3,:) + temp;

            %rfe
           % ranking = spider_wrapper(X_train,Y_train,numF,lower('rfe'));
            %temp = get_rate(X_train,Y_train,X_test, Y_test,numF,P,ranking);
            %tab_res(5,:) = tab_res(5,:) + temp;

            %L0
            ranking = spider_wrapper(X_train,Y_train,numF,lower('l0'));
            temp = get_rate(X_train,Y_train,X_test, Y_test,numF,P,ranking);
            tab_res(4,:) = tab_res(4,:) + temp;

            %fisher
            %ranking = spider_wrapper(X_train,Y_train,numF,lower('fisher'));
            %temp = get_rate(X_train,Y_train,X_test, Y_test,numF,P,ranking);
            %tab_res(7,:) = tab_res(7,:) + temp;

            %UDFS


            %nested cross validation (2)
            max_rate=0;
            nClass_max=0;
            for k=10:10:50
                acc_final=0;
                for fold_ncv = 1:5
                    X_train_ncv = new_X(training(P_NCV,fold_ncv),:);
                    Y_train_ncv = new_Y(training(P_NCV,fold_ncv),:);
                    X_test_ncv = new_X(P_NCV.test(fold_ncv),:);
                    Y_test_ncv = new_Y(P_NCV.test(fold_ncv),:);
                    ranking = UDFS(X_train_ncv , k ); 
                    acc_tmp = ncv(X_train_ncv,Y_train_ncv,X_test_ncv, Y_test_ncv,numF,P_NCV,ranking);
                    acc_final = acc_final + acc_tmp;
                end 
                acc_final= acc_final / 5;
                if acc_final > max_rate
                    max_rate=acc_final;
                    nClass_max = k;
                end
            end

            ranking = UDFS(X_train , nClass_max ); 
            temp = get_rate(X_train,Y_train,X_test, Y_test,numF,P_NCV,ranking);
            tab_res(5,:) = tab_res(5,:) + temp;

            %cfs
            ranking = cfs(X_train);
            temp = get_rate(X_train,Y_train,X_test, Y_test,numF,P,ranking);
            tab_res(6,:) = tab_res(6,:) + temp;

            %llcfs
            ranking = llcfs( X_train );
            temp = get_rate(X_train,Y_train,X_test, Y_test,numF,P,ranking);
            tab_res(7,:) = tab_res(7,:) + temp;

        end    
    end
    array = 10:10:100;
    tab_res=tab_res/sum(c.TrainSize);
    sTable = vertcat(array,tab_res);

end


function res = get_rate(X_train,Y_train,X_test, Y_test,numF,P,ranking)

	res = [];
    options=statset;
    options.MaxIter=100000;
    for k = 10:10:100
	    % Use a linear support vector machine classifier
	    svmStruct = svmtrain(X_train(:,ranking(1:k)),Y_train,'showplot',true,'options',options);
	    C = svmclassify(svmStruct,X_test(:,ranking(1:k)),'showplot',true);
	    err_rate = sum(Y_test~= C); % mis-classification rate 
	    res = [res, 100*(1-err_rate)];
    end 
end

function err_rate = ncv(X_train,Y_train,X_test, Y_test,numF,P,ranking)
	options=statset;
    options.MaxIter=1000000;
    max_err_rate=0;
    best_k=0;
    for k=10:10:100 
	    % Use a linear support vector machine classifier
	    svmStruct = svmtrain(X_train(:,ranking(1:k)),Y_train,'showplot',true,'options',options);
	    C = svmclassify(svmStruct,X_test(:,ranking(1:k)),'showplot',true);
	    err_rate = sum(Y_test~= C); % mis-classification rate 
         if err_rate > max_err_rate
                max_err_rate=err_rate;
                best_k = k;
         end
    end
    svmStruct = svmtrain(X_train(:,ranking(1:best_k)),Y_train,'showplot',true,'options',options);
	C = svmclassify(svmStruct,X_test(:,ranking(1:best_k)),'showplot',true);
    err_rate = sum(Y_test~= C); % mis-classification rate 
        
end