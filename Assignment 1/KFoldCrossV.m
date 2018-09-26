function [avgTrainingError,avgValidError] = KFoldCrossV(x, y, k, d)

data=[x,y];
% test_data=data(1:size(data,1)/5,:);
% data=data((size(data,1)/5)+1:size(data,1),:);
%get index of each fold
indices=crossvalind('Kfold',data(1:size(data,1),size(data,2)),5);

avgTrainingError=zeros(d, 1);
avgValidError=zeros(d, 1);
%avgTestError=zeros(15, 1);

for degree = 1:d
    
ValidError = zeros(k, 1);
TrainingError = zeros(k, 1);
% TestError = zeros(5, 1);
    for i=1:k
        %set the k fold index as 1 and the others as 0
        validIndex = (indices == i);
        trainIndex = ~validIndex;
        train_data=data(trainIndex,:);
        valid_data=data(validIndex,:);

         [X1,w]= PolyRegress(train_data(:,1), train_data(:,2), degree);   
         TrainingError(i) =  trainingErrorEvaluate(X1, train_data(:,2), w);

         X = [valid_data(:,1), ones(length(valid_data(:,1)), 1)];
         % adds x2...xd to the x
         for power = 2:degree
            X = [ valid_data(:,1).^power, X];
         end
         ValidError(i) =  trainingErrorEvaluate(X, valid_data(:,2), w);
%          [X3,s]= PolyRegress(test_data(:,1), test_data(:,2), degree);
%          TestError(k)=trainingErrorEvaluate(X3, test_data(:,2), w);
    end
    avgTrainingError(degree)=mean(TrainingError);
    avgValidError(degree)=mean(ValidError);
%     avgTestError(degree)=mean(TestError);
end

figure
plot(avgTrainingError,'LineWidth',1.5);
hold on;
plot(avgValidError,'LineWidth',1.5);
ylim([0 45]);
legend('Training Error','Valid Error','location','best');
hold off