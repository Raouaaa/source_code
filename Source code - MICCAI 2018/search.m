function feat_mat = search(data,ranking,com1,com2)

    extr_data = extract_subject(data); %35x35
    X = importdata(data);
     
    avg_X = zeros(1,595);
    for i =1:77
        avg_X = avg_X + X(i,:);
    end
    avg_X=avg_X/77;
    
    avg_X = avg_X(ranking);
    
    temp_mat=zeros(35,35);
    
    %create test
    numb = zeros(35,35);
    lim=0;j=1;k=1;
    for i=2:35
        numb(j:j+lim,i)=k:k+lim;
        k=max(numb(:))+1;
        lim=lim+1;
    end
    feat_mat = zeros(10,3);
    for i=1:10
        
        test = (ranking(i)==numb);
        [maxNum, maxIndex] = max(test(:));
        [row, col] = ind2sub(size(test), maxIndex);
        temp_mat(row,col) = avg_X(i);
        temp_mat(col,row) = avg_X(i);
        feat_mat(i,1)=row;
        feat_mat(i,2)=col;
        feat_mat(i,3)=avg_X(i);
    end
   
    
    
    norm_data = normalise(temp_mat);
    
    %defaultLabel = cell(10);
    %for i =1:10
    %  defaultLabel{i} = num2str(ranking(i));
    %end
    
    figure
    circularGraph(norm_data,'Colormap',brewermap(size(norm_data,1),'*Spectral'))
    text(1,-1,strcat('Best FS method:  ',com1))
    text(1,-1.1,strcat('Avg accuracy : ',com2))

end

function norm_mat = normalise(matrix)
    norm_mat = zeros(size(matrix,1),size(matrix,2));
    for i=1:size(matrix,1)
        for j=1:size(matrix,2)
            norm_mat(i,j) = 100*(matrix(i,j) - min(matrix(:)) ) / ( max(matrix(:)) - min(matrix(:)) );
        end
    end
end