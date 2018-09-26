%% c)
x = load('hw1x.dat');
y = load('hw1y.dat');

% set diagonal matrix all to 1
U = diag(ones(1, length(x)));
% find the largest and its index 
[maxVal, maxIndex] = max(y);

% add a 1's column vector 
vecx = [x, ones(length(x), 1)];

% set the weight range
maxValWeighted = ([1,5,10,15,20]);

% plot the line to show the change
figure
scatter(x, y,'filled','LineWidth',1);
hold on
for n=1:size(maxValWeighted,2)
%weighted as i
U(maxIndex(1), maxIndex(1)) = maxValWeighted(n);
w = WeightedLinearRegress( x, y, U);
plot(x, vecx*w);
ylim([-2 2.5]);
end
legend('training data','Weighted 1','Weighted 5','Weighted 10','Weighted 15','Weighted 20','location','best');
hold off
%% d)
% set diagonal matrix all to 1
U = diag(zeros(1, length(x)));

figure
% plot the training data
scatter(x, y,'filled','LineWidth',1);
hold on
% plot the linear regression line
w = (vecx'*vecx)^-1 * vecx' * y;
line=vecx*w;
plot(x,line,'LineWidth',1.5);
%set a matrix in input range 
dot_x=linspace(min(x),max(x),100);
dot_y=zeros(length(dot_x),1);
%for each input point, define the weight according to the distance to every
%training data point 
for m=1:1:length(dot_x)
    for n=1:1:length(x)
        %using the Gaussian distribution to set the weight, the training
        %point which is closest to the dot_x(m) get hightest weight, the
        %variance set as 1
        U(n,n)=exp(-((dot_x(m)-vecx(n,1))^2)/(2*1));
    end
    w= WeightedLinearRegress( x, y, U);
dot_y(m)=w(1)*dot_x(m)+w(2);
end
%so, in the end, every point of the line close the training point which is
%closest to the current point, we can get a fitting line with using a linear 
%regression.
plot(dot_x,dot_y,'LineWidth',1.5);
ylim([-2 2.5]);
legend('training data','linear regression','Weighted linear regression','location','best');
hold off




