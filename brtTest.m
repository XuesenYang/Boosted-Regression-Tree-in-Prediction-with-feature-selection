function output = brtTest( input, brtModel, varargin )
    %input是预测的输入，一个样本D维度
    %brtModel是训练好的模型
    %varargin 默认的树数
    if isempty(varargin)
        len = length(brtModel)-1;
    else        
        len = varargin{1};
    end
            
    nu = brtModel{end};
    output = brtModel{1};
    
    for i=2:len
        output = output + nu * regressionTreeTest( input, brtModel{i} );
    end
end