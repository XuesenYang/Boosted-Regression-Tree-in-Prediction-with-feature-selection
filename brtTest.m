function output = brtTest( input, brtModel, varargin )
    %input��Ԥ������룬һ������Dά��
    %brtModel��ѵ���õ�ģ��
    %varargin Ĭ�ϵ�����
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