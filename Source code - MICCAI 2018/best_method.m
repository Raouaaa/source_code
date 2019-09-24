% The overall function, returns an overview of the data
% It calls the other functions to create the results' table, select the
% best FS method and plot the different graphs

function C=best_method(data)
     %Remove warnings
    warning('off', 'stats:obsolete:ReplaceThisWithMethodOfObjectReturnedBy');
    warning('off', 'stats:obsolete:ReplaceThisWith');
    warning('off', 'stats:svmclassify:NoTrainingFigure');
    warning('off', 'stats:svmtrain:OnlyPlot2D');
    warning('off', 'MATLAB:singularMatrix');
    warning('off', 'MATLAB:illConditionedMatrix');
    warning('off', 'MATLAB:nearlySingularMatrix');
    
    
    tab_acc = create_tab(data);
  
    res1=0;
    res2=0;
    res3=0;
    ranking1 = 0;
    ranking2 = 0;
    ranking3 = 0;
    [tensor_avg idxx] = create_graph(data,tab_acc);
   
    %prepare SVM
    X = importdata(data);
    labels = importdata('labels.mat');
    Y = labels;
   
    numF = size(X,2);
    sum_res=0;
    ranking =  get_ranking(data,idxx);

    sum_res = tab_acc(idxx +1,:)
    
    listFS = {'relieff','mutinffs','laplacian','L0','UDFS','llcfs','cfs'};
    %sum_res = sum_res/77
    x_axis = 10:10:100;
   [max_k idxx_k] = max(sum_res);
    figure
    plot(x_axis,sum_res);
    title(data);
    hold on;
    plot(idxx_k*10,max_k,'*','MarkerSize',10);
   
    %plot CircGraph
     mk = num2str(max_k);
    search(data,ranking(1:10),listFS{idxx},strcat(mk(1:5),'%') );
    
    %extr_data = extract_subject(data);
    %norm_data = normalise(extr_data(ranking(1:10),ranking(1:10))); % 595 or 35x35
    %defaultLabel = cell(10);
    %for i =1:10
    %  defaultLabel{i} = num2str(ranking(i));
    %end
    
    %figure
    %circularGraph(norm_data,'Label',defaultLabel,'Colormap',brewermap(size(norm_data,1),'*Spectral'))
    %text(1,-1,strcat('Best FS method:  ',listFS{idxx}))
    %text(1,-1.1,strcat('Max accuracy : ',num2str(max_k/100)))
   
    
    %find 3 methods closest to the best one 
    index_method= zeros(7,2);
    index_method(:,1)=1:7;
    index_method(:,2)=tensor_avg(:,idxx)
    
    [~,idx] = sort(index_method(:,2),'descend');
    index_method = index_method(idx,:)
    %index_method = index_method(1:3,:); % take top3
    
    %Run SVM for the 3 methods for k = 10 to 100
    ranking1 = get_ranking(data,index_method(1,1));
    ranking2 = get_ranking(data,index_method(2,1));
    ranking3 = get_ranking(data,index_method(3,1));
    c = cvpartition(Y,'KFold',5);
    for kfold = 1:5
        
        new_Y = Y(c.training(kfold));
        new_X = X(c.training(kfold),:);
        for fold = 1:c.TrainSize(kfold)
            P = cvpartition(new_Y,'LeaveOut'); % Leave One out
            X_train = new_X(training(P,fold),:);
            Y_train = new_Y(training(P,fold),:);
            X_test = new_X(P.test(fold),:);
            Y_test = new_Y(P.test(fold),:);

            res1 = res1 + get_rate(X_train,Y_train,X_test, Y_test,ranking1,idxx_k);
            res2 = res2 + get_rate(X_train,Y_train,X_test, Y_test,ranking2,idxx_k);
            res3 = res3 + get_rate(X_train,Y_train,X_test, Y_test,ranking3,idxx_k);

        end
    end
    res1 = res1/77/5;
    res2 = res2/77/5;
    res3 = res3/77/5;
    
   
    C = cell(4,5);
    C{1,1} = strcat('Best FS method:  ',listFS{idxx});
    C{1,3} = strcat('Max accuracy : ',mk(1:5),'%');
     C{1,4} = strcat('Average accuracy : ',num2str(mean(sum_res)));
    C{1,5} = strcat('Best Number of Features : ',num2str(idxx_k*10));
    C{2,1}= 'FS methods';
    C{2,2}= 'Accuracy';
    C{2,3}= 'Overlap top 50 (in %)';
    C{2,4}= 'Overlap top 20 (in %)';
    C{2,5}= 'Overlap top 15 (in %)';
    
    C{3,1}=listFS{index_method(1,1)};
    C{3,2}=res1;
    C{3,3}=get_score(ranking,ranking1,50);
    C{3,4}=get_score(ranking,ranking1,20);
    C{3,5}=get_score(ranking,ranking1,15);
    C{4,1}=listFS{index_method(2,1)};
    C{4,2}=res2;
    C{4,3}=get_score(ranking,ranking2,50);
    C{4,4}=get_score(ranking,ranking2,20);
    C{4,5}=get_score(ranking,ranking2,15);
    C{5,1}=listFS{index_method(3,1)};
    C{5,2}=res3;
    C{5,3}=get_score(ranking,ranking3,50);
    C{5,4}=get_score(ranking,ranking3,20);
    C{5,5}=get_score(ranking,ranking3,15);

end


function res_err = get_rate(X_train,Y_train,X_test, Y_test,ranking,k)

    res_err=0;
    options=statset;
    options.MaxIter=100000;
    for k = 10:10:100
        % Use a linear support vector machine classifier
        svmStruct = svmtrain(X_train(:,ranking(1:k)),Y_train,'showplot',true,'options',options);
        C = svmclassify(svmStruct,X_test(:,ranking(1:k)),'showplot',true);
        err_rate = sum(Y_test~= C); % mis-classification rate 
        res_err = res_err + err_rate;
    end  
    
    res_err = res_err/10;
end

function score = get_score(mat1,mat2,top)
    score=0;
    nmat1 = mat1(1:top,:);
    nmat2 = mat2(1:top,:);
    for i = 1:top
            for j=1:top
                if mat1(i,1) == mat2(j,1)
                    score=score+1;
                end
            end
        end
score=100*score/top;
end

function norm_mat = normalise(matrix)
    norm_mat = zeros(size(matrix,1),size(matrix,2));
    for i=1:size(matrix,1)
        for j=1:size(matrix,2)
            norm_mat(i,j) = 100*(matrix(i,j) - min(matrix(:)) ) / ( max(matrix(:)) - min(matrix(:)) );
        end
    end
end
