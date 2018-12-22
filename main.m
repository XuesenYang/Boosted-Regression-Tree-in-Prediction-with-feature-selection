% function err=main(leafNum,treeNum,nu,varargin)
traindata=load('zhengqi_train.txt');
[a,b]=size(traindata);
train_data=traindata(1:round(0.8*a),:);
verify_data=traindata(round(0.8*a)+1:end,:);
test_data=load('zhengqi_test.txt');
leafNum=2;
treeNum=600;
nu=0.05;
varargin=600;
%leafNum¡¾1£¬5¡¿£¬treeNum¡¾50£¬500¡¿£¬nu¡¾0.001£¬0.05¡¿£¬varargin¡¾50£¬treeNum¡¿
% leafNum=pop(i,1);
% treeNum=pop(i,2);
% nu=pop(i,3);
% varargin=pop(i,4);
n_d=round(0.7*b);
n_p=50;
rang_l=1;
rang_r=size(traindata,2)-1;
%%  initialize
n_iteration = 0;
max_iteration=40;
popu = rang_l + (rang_r - rang_l) * rand(n_p,n_d);
VStep =rand(n_p,n_d);
best_fitness = 1000000*ones(max_iteration,1);
fitness_popu = 1000000*ones(n_p,1);  % store fitness value for each individual
    for idx=1:n_p
        popu111=floor(popu(idx,:));
        popu1=unique(popu111);
        popu1length=length(popu1);
        popu2=setdiff(1:rang_r,popu1);
        randIndex = randperm(size(popu2,2));
        popu3=popu2(1,randIndex);
        val=[popu1 popu3(1,1:n_d-popu1length)];
        brtModel=brtTrain( train_data(:,val), train_data(:,end), leafNum, treeNum, nu );
        for i=1:(a-round(0.8*a))
            output(i,1)=brtTest( verify_data(i,val), brtModel, varargin );
        end
        err(idx) = immse(output,verify_data(:,end));
    end
    fitness_popu=err;
    PBest = popu;
    fPBest =fitness_popu;
    [fGBest, g] = min(fPBest);
    GBest = popu(g,:);
    c1_now= 2.0;
    c2_now= 2.0;
    w_start=0.9;
    w_end=0.4;
    vmax=2;
%% looping
while n_iteration < max_iteration
         indi_temp=popu;
         w_now = ((w_start-w_end)*(max_iteration-n_iteration)/max_iteration)+w_end;
         R1 = rand(n_p,n_d);
         R2 = rand(n_p,n_d);
         A= repmat(indi_temp(g,:), n_p, 1);
         VStep =  w_now*VStep + c1_now*R1.*(PBest-indi_temp) + c2_now*R2.*(A-indi_temp);
         changeRows = VStep > vmax;
         VStep(find(changeRows)) =vmax;
         changeRows = VStep < -vmax;
         VStep(find(changeRows)) = -vmax;
         indi_temp=indi_temp+VStep;
         changeRow = indi_temp > rang_r;
         indi_temp(find(changeRow)) =rang_r;
         changeRows = indi_temp < rang_l;
         indi_temp(find(changeRow)) =rang_l;
  for idx=1:n_p
        indi_temp111=round(indi_temp(idx,:));
        changeRows1 = indi_temp111<=0;
        indi_temp111(changeRows1)=1;
        changeRows2 = indi_temp111>rang_r;
        indi_temp111(changeRows2)=rang_r;
        indi_temp1=unique(indi_temp111);
        indi_temp1length=length(indi_temp1);
        indi_temp2=setdiff(1:rang_r,indi_temp1);
        randIndex = randperm(size(indi_temp2,2));
        indi_temp3=indi_temp2(1,randIndex);
        val=[indi_temp1 indi_temp3(1,1:n_d-indi_temp1length)];
        vol(idx,:)=val;
        brtModel=brtTrain( train_data(:,val), train_data(:,end), leafNum, treeNum, nu );
        for i=1:(a-round(0.8*a))
            output(i,1)=brtTest( verify_data(i,val), brtModel, varargin );
        end
        err(idx) = immse(output,verify_data(:,end));
        fv = err(idx);
        if fv < fitness_popu(idx)  % better than the previous one, replace
            fitness_popu(idx) = fv;
            popu(idx,:) = indi_temp(idx,:);
        end 
  end
    n_iteration = n_iteration +1;
    PBest = popu;
    fPBest =fitness_popu;
    [fGBest, g] = min(fPBest);
    GBest = popu(g,:);
    volbest=vol(g,:);
    fprintf('subsetsize: %d \t Iter: %d \t Err: %.4f \t \n',n_d,n_iteration,fGBest)
end   
 %% Forecast test_data:file name--'zhengqi_test.txt'                    
for j=1:length(test_data)
    result(j,1)=brtTest( test_data(j,volbest), brtModel, varargin );
end
%% save result
dlmwrite('predict.txt',result);