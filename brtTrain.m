function brtModel = brtTrain( X, T, leafNum, treeNum, nu )
%BRT Summary of this function goes here
%   Detailed explanation goes here
    % X is the input data, N samples with D dimension
    % T is the target matrix, N samples K dimension
    % LeafNum is the number of leaf nodes of regression tree
    % treeNum is the Number of binary regression trees will add to the model
    % Nu is a shrinkage parameter
    % if you can not Call c++ Script:findBestSplit.cpp, run command in Command window:mex findBestSplit.cpp COMPFLAGS="/openmp $COMPFLAGS"
    brtModel = cell(treeNum+1,1);
    
    brtModel{1} = mean(T);
    brtModel{end} = nu;
    estT = zeros( size(T) ); % estimated T from current tree
    previousT = repmat( brtModel{1}, size(T,1), 1 ); % estimated T from current brt
    residualT = zeros( size(T) );
    
    for i=2:treeNum
%         fprintf( 'tree num = %d\n', i );
        residualT = T - previousT;
        brtModel{i} = regressionTreeTrain( X, residualT, leafNum );
        
        for j=1:size(T,1)
            estT(j,:) = regressionTreeTest( X(j,:), brtModel{i} );
        end
        previousT = previousT + nu * estT;
    end
end

