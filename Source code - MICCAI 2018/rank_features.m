%Run a specific FS method and return a matrix with the index of the feature
%and its average score. The matrix is ordered by increasing score
function index_rank = rank_features(data)

    clc;
    rng(1);
    listFS = {'relieff','mutinffs','fsv','laplacian','rfe','L0','fisher','UDFS','llcfs','cfs'};

    [ methodID ] = readInput( listFS );
    selection_method = listFS{methodID}; % Selected
    

    X = importdata(data);
    labels = importdata('labels.mat');
	   Y = labels;
  	% number of features
  	numF = 595;
    mat_temp = zeros(595,1);
      
      P = cvpartition(Y,'LeaveOut');
   	% feature Selection on training data
  switch lower(selection_method)
   
    
    case 'relieff'
   for i = 1:77
        X_1 = X(P.training(i),:);
        Y_1 = Y(P.training(i),:);
        
        max_rate=0;
        best_k=5;
        P_NCV = cvpartition(Y_1,'KFold',5);
        for k=5:5:50
            acc_final=0;
            for fold_ncv = 1:5
                X_train_ncv = X_1(training(P_NCV,fold_ncv),:);
                Y_train_ncv = Y_1(training(P_NCV,fold_ncv),:);
                X_test_ncv = X_1(P_NCV.test(fold_ncv),:);
                Y_test_ncv = Y_1(P_NCV.test(fold_ncv),:);
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
        
        [ranking, w] = reliefF( X_1, Y_1, best_k);
        mat_temp(:,1) = mat_temp(:,1) + transpose(ranking);
    end
         
    case 'mutinffs'
        for i = 1:77
            X_1 = X(P.test(i),:);
        Y_1 = Y(P.test(i),:);
           [ ranking , w] = mutInfFS(X_1, Y_1, numF );
            mat_temp(:,1) = mat_temp(:,1) + ranking;
       end
        
    case 'fsv'
        for i = 1:77
            X_1 = X(P.test(i),:);
            Y_1 = Y(P.test(i),:);
            [ ranking , w] = fsvFS( X_1, Y_1, numF );
            mat_temp(:,1) = mat_temp(:,1) + ranking;
        end
        
    case 'laplacian'
                   
        for i = 1:77
            X_1 = X(P.training(i),:);
            Y_1 = Y(P.training(i),:);
            W = dist(X_1');
            W = -W./max(max(W)); % it's a similarity
            [lscores] = LaplacianScore(X_1, W);
            [junk, ranking] = sort(-lscores);
            mat_temp(:,1) = mat_temp(:,1) + ranking;
        end
        
    case 'rfe'
        for i = 1:77
             X_1 = X(P.test(i),:);
        Y_1 = Y(P.test(i),:);
            ranking = spider_wrapper( X_1, Y_1,numF,lower(selection_method));
            mat_temp(:,1) = mat_temp(:,1) + ranking';
        end
        
    case 'l0'
        
        for i = 1:77
             X_1 = X(P.test(i),:);
        Y_1 = Y(P.test(i),:);
            ranking = spider_wrapper(X_1,Y_1,numF,lower(selection_method));
            mat_temp(:,1) = mat_temp(:,1) + ranking';
        end
        
    case 'fisher'
         for i = 1:77
            X_1 = X(P.training(i),:);
            Y_1 = Y(P.training(i),:);
            ranking = spider_wrapper(X_1,Y_1,numF,lower(selection_method));
            mat_temp(:,1) = mat_temp(:,1) + ranking';
        end
                       
    case 'udfs'
        % Regularized Discriminative Feature Selection for Unsupervised Learning
                  
        
       for i = 1:77
        X_1 = X(P.training(i),:);
        Y_1 = Y(P.training(i),:);
        
        max_rate=0;
        P_NCV = cvpartition(Y_1,'KFold',5);
        nClass_max=0;
        for k=5:5:50
            acc_final=0;
            for fold_ncv = 1:5
                 X_train_ncv = X_1(training(P_NCV,fold_ncv),:);
                Y_train_ncv = Y_1(training(P_NCV,fold_ncv),:);
                X_test_ncv = X_1(P_NCV.test(fold_ncv),:);
                Y_test_ncv = Y_1(P_NCV.test(fold_ncv),:);
                ranking = UDFS(X_train_ncv , k); 
                acc_tmp = ncv(X_train_ncv,Y_train_ncv,X_test_ncv, Y_test_ncv,numF,P_NCV,ranking);
                acc_final = acc_final + acc_tmp;
            end 
            acc_final= acc_final / 5;
            if acc_final > max_rate
                max_rate=acc_final;
                nClass_max = k;
            end
        end
           
           ranking = UDFS(X_1 , nClass_max ); 
            mat_temp(:,1) = mat_temp(:,1) + ranking;
       end
        
    case 'cfs'
        % BASELINE - Sort features according to pairwise correlations
        
       for i = 1:77
        X_1 = X(P.training(i),:);
        ranking = cfs(X_1);  
        mat_temp(:,1) = mat_temp(:,1) + ranking;
       end
        
        
    case 'llcfs'   
        % Feature Selection and Kernel Learning for Local Learning-Based Clustering
        
     for i = 1:77
        X_1 = X(P.training(i),:);
        ranking = llcfs(X_1);
        mat_temp(:,1) = mat_temp(:,1) + ranking;
     end
        
    otherwise
        disp('Unknown method.')
end
   
    
    index_rank= zeros(595,2);
    index_rank(:,1)=1:595;
    %mat_temp
    index_rank(:,2)=mat_temp/77;
    [~,idx] = sort(index_rank(:,2));
    index_rank = index_rank(idx,:);
end

function err_rate = ncv(X_train,Y_train,X_test, Y_test,numF,P,ranking)
	options=statset;
    options.MaxIter=1000000;
    k=10;   
	    % Use a linear support vector machine classifier
	    svmStruct = svmtrain(X_train(:,ranking(1:k)),Y_train,'showplot',true,'options',options);
	    C = svmclassify(svmStruct,X_test(:,ranking(1:k)),'showplot',true);
	    err_rate = sum(Y_test~= C); % mis-classification rate 
        
end