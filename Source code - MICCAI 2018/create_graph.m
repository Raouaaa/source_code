%Run the rank_feature method for a specific data set
%It returns the matrix of similarity like that:
%     FS1  FS2  FS3
% FS1  0    %    %
% FS2  %    0    %
% FS3  %    %    0
% And the index of the best FS method in this list: {'relieff','mutinffs','laplacian','L0','UDFS','llcfs','cfs'}

%idx = index of the best method
function  [tensor_avg n_idx] = create_graph(data,tab_acc)

 %Remove warnings
	warning('off', 'stats:obsolete:ReplaceThisWithMethodOfObjectReturnedBy');
	warning('off', 'stats:obsolete:ReplaceThisWith');
    warning('off', 'stats:svmclassify:NoTrainingFigure');
    warning('off', 'stats:svmtrain:OnlyPlot2D');
    warning('off', 'MATLAB:singularMatrix');
    warning('off', 'MATLAB:illConditionedMatrix');
    warning('off', 'MATLAB:nearlySingularMatrix');
    
    rng(1);
    mat_graph = zeros(7,7,10);
    mat1 = rank_features_graph(data,'relieff');
    mat2 = rank_features_graph(data,'mutinffs');
   
    mat3 = rank_features_graph(data,'laplacian');
  
    mat4 = rank_features_graph(data,'l0');
    
    mat5 = rank_features_graph(data,'UDFS');
    mat6 = rank_features_graph(data,'llcfs');
    mat7 = rank_features_graph(data,'cfs');
    for top=10:10:100
        %1
        mat_graph(1,2,top/10)= get_score(mat1,mat2,top);
        mat_graph(1,3,top/10)= get_score(mat1,mat3,top);
        mat_graph(1,4,top/10)= get_score(mat1,mat4,top);
        mat_graph(1,5,top/10)= get_score(mat1,mat5,top);
        mat_graph(1,6,top/10)= get_score(mat1,mat6,top);
        mat_graph(1,7,top/10)= get_score(mat1,mat7,top);
        %2
        mat_graph(2,3,top/10)= get_score(mat2,mat3,top);
        mat_graph(2,4,top/10)= get_score(mat2,mat4,top);
        mat_graph(2,5,top/10)= get_score(mat2,mat5,top);
        mat_graph(2,6,top/10)= get_score(mat2,mat6,top);
        mat_graph(2,7,top/10)= get_score(mat2,mat7,top);
        %3
        mat_graph(3,4,top/10)= get_score(mat3,mat4,top);
        mat_graph(3,5,top/10)= get_score(mat3,mat5,top);
        mat_graph(3,6,top/10)= get_score(mat3,mat6,top);
        mat_graph(3,7,top/10)= get_score(mat3,mat7,top);
        %4
        mat_graph(4,5,top/10)= get_score(mat4,mat5,top);
        mat_graph(4,6,top/10)= get_score(mat4,mat6,top);
        mat_graph(4,7,top/10)= get_score(mat4,mat7,top);
        %5
        mat_graph(5,6,top/10)= get_score(mat5,mat6,top);
        mat_graph(5,7,top/10)= get_score(mat5,mat7,top);
        %6
        mat_graph(6,7,top/10)= get_score(mat6,mat7,top);   
    end


    %fill all the matrices
    for i=1:10
        for j=1:7
            for k=1:7
                mat_graph(k,j,i)=mat_graph(j,k,i);
            end
        end
    end

    %subplot all the views
    figure
    for i=1:10
        subplot(5,2,i)
        imagesc(mat_graph(:,:,i), [0 100])
        colormap(brewermap([],'*Spectral')) 
        colorbar
        title(strcat(data(1:2),'emisphere - view', data(6),'- top ', int2str(i*10),' features'));
    end

    tensor_avg=zeros(7,7);
    tensor_avg = avg_tensor(mat_graph);
    figure
    imagesc(tensor_avg, [0 100])
    colormap(brewermap([],'*Spectral')) 
    title(strcat(data(1:2),'emisphere - view', data(6),' similarity'))
    colorbar
    n_acc=0;
  
    
    %Graph_t = graph(tensor_avg,{'relieff','mutinffs','laplacian','L0','UDFS','llcfs','cfs'});
    %degree_I = centrality(Graph_t,'degree','Importance',Graph_t.Edges.Weight);
    %[max_d, idx] = max(degree_I);
    
    %LWidths = 4*Graph_t.Edges.Weight/max(Graph_t.Edges.Weight);
    %figure
    %H= plot(Graph_t,'LineWidth',LWidths);
    %set(gca,'XTick',[]);
    %set(gca,'YTick',[]);
    %highlight(H,idx,'NodeColor','b','Marker','h','MarkerSize',10);
    %title(data)    
    
    
   
    
    nouv_tens = zeros(7,7);
    norm_tens = zeros(7,7);
    %Before Building graphs, * avg simil matrix by average accuracy
    % and plot the graph that result out of it
    res_tab= [];
    for i = 2:8
        for j=1:10
             n_acc= n_acc + tab_acc(i,j);
        end
        res_tab(i-1) = n_acc/10;
        n_acc=0;
    end
    
    %res_tab = normalise(res_tab);
    %norm_tens = normalise(tensor_avg);
    
    for i = 1:7
        for j=1:7
            norm_res(i,j) = abs(res_tab(i)-res_tab(j));
        end
    end
    
    norm_res = normalise(norm_res);
    figure
    imagesc(norm_res, [0 100])
    colormap(brewermap([],'*Spectral'))
    colorbar
    title('Differences of acc')
    
    % TO ADD STABILITY uncomment next line
    stab_mat = get_stability(data);
    % TO REMOVE STABILITY uncomment next line
    %stab_mat = ones(7,7);

      for i = 1:7
        for j=1:7
           nouv_tens(i,j) = norm_res(i,j)*tensor_avg(i,j)*stab_mat(i,j);
        end
      end
    nouv_tens = normalise(nouv_tens)
    figure
    imagesc(nouv_tens, [0 100])
    colormap(brewermap([],'*Spectral'))
    colorbar
    title('Final')
   
    
    
    nouv_graph = graph(nouv_tens,{'relieff','mutinffs','laplacian','L0','UDFS','llcfs','cfs'},'upper');
    n_degree_I = centrality(nouv_graph,'degree','Importance',nouv_graph.Edges.Weight);
    [n_max_d,n_idx] = max(n_degree_I);
    
    n_LWidths = 4*nouv_graph.Edges.Weight/max(nouv_graph.Edges.Weight);
    figure
    n_H= plot(nouv_graph,'-o','LineWidth',n_LWidths);
    set(gca,'XTick',[]);
    set(gca,'YTick',[]);
    highlight(n_H,n_idx,'NodeColor','b','Marker','h','MarkerSize',10);
    title(strcat(data,' test'))    
    
    % circular graph for top 5 10 15    
    
    mat_graph = zeros(7,7);
    for top=5:5:15
        %1
        mat_graph(1,2)= get_score(mat1,mat2,top);
        mat_graph(1,3)= get_score(mat1,mat3,top);
        mat_graph(1,4)= get_score(mat1,mat4,top);
        mat_graph(1,5)= get_score(mat1,mat5,top);
        mat_graph(1,6)= get_score(mat1,mat6,top);
        mat_graph(1,7)= get_score(mat1,mat7,top);
        %2
        mat_graph(2,3)= get_score(mat2,mat3,top);
        mat_graph(2,4)= get_score(mat2,mat4,top);
        mat_graph(2,5)= get_score(mat2,mat5,top);
        mat_graph(2,6)= get_score(mat2,mat6,top);
        mat_graph(2,7)= get_score(mat2,mat7,top);
        %3
        mat_graph(3,4)= get_score(mat3,mat4,top);
        mat_graph(3,5)= get_score(mat3,mat5,top);
        mat_graph(3,6)= get_score(mat3,mat6,top);
        mat_graph(3,7)= get_score(mat3,mat7,top);
        %4
        mat_graph(4,5)= get_score(mat4,mat5,top);
        mat_graph(4,6)= get_score(mat4,mat6,top);
        mat_graph(4,7)= get_score(mat4,mat7,top);
        %5
        mat_graph(5,6)= get_score(mat5,mat6,top);
        mat_graph(5,7)= get_score(mat5,mat7,top);
        %6
        mat_graph(6,7)= get_score(mat6,mat7,top);   
   
        for j=1:7
         for k=1:7
            mat_graph(k,j)=mat_graph(j,k);
          end
         end
     figure
    circularGraph(mat_graph);
    title(data)  
    end
    
   
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

function tensor_avg = avg_tensor(tensor)
    tensor_avg=zeros(7,7);
    for i=1:size(tensor,3)
        for j=1:size(tensor,2)
            for k=1:size(tensor,1)
                tensor_avg(k,j)=tensor_avg(k,j)+tensor(k,j,i);
            end
        end
    end
    tensor_avg=tensor_avg/10;
end


function norm_mat = normalise(matrix)
    norm_mat = zeros(size(matrix,1),size(matrix,2));
    for i=1:size(matrix,1)
        for j=1:size(matrix,2)
            norm_mat(i,j) = 100*(matrix(i,j) - min(matrix(:)) ) / ( max(matrix(:)) - min(matrix(:)) );
        end
    end
end