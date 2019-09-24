function norm_mat = normalise(matrix)
    norm_mat = zeros(size(matrix,1),size(matrix,2));
    for i=1:size(matrix,1)
        for j=1:size(matrix,2)
            norm_mat(i,j) = 100*(matrix(i,j) - min(matrix(:)) ) / ( max(matrix(:)) - min(matrix(:)) );
        end
    end
end