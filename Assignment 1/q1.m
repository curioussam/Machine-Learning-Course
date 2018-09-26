
%%Question 1
%% a)load the data into memory and plot it
x = load('hw1x.dat');
y = load('hw1y.dat');

scatter(x, y,'filled','LineWidth',1);
legend('training data','location','best');

%% b) add a colum vector
vecx = [x, ones(length(x), 1)];

%linear regression formula
w = (vecx'*vecx)^-1 * vecx' * y;
disp(w)

%plot the linear regression line
figure
line=vecx*w;
plot(x, y,'.',x,line,'LineWidth',1.5);
legend('training data', 'linear regression','location','best');

%% c? Evaluate Function
E= trainingErrorEvaluate( vecx, y, w );
fprintf( 'The training error is: %f\n', E);

%% d? PolyRegress(x,y,d) Function
[x_Poly,w_Poly]  = PolyRegress( x, y, 2 );
disp(w_Poly)

%% e)plot it
E= trainingErrorEvaluate( x_Poly, y, w_Poly );
fprintf( 'The training error is: %f\n', E);

figure
x_dot = linspace(min(x),max(x));
line_Poly2=polyval(w_Poly, x_dot);
plot(x, y,'.',x_dot,line_Poly2);
legend('training data', 'polynomial regression','location','best');

%% f)PolyRegress(x,y,3)
[x_Poly,w_Poly]  = PolyRegress( x, y, 3);
disp(w_Poly)

E= trainingErrorEvaluate( x_Poly, y, w_Poly );
fprintf( 'The training error is: %f\n', E);

figure
line_Poly3=polyval(w_Poly, x_dot);
plot(x, y,'.',x_dot,line_Poly2,x_dot,line_Poly3,'g');
legend('training data', 'quadratic regression','cubic regression','location','best');

%% h)five-fold cross validation 

[TrainingError,ValidError]= KFoldCrossV(x, y, 5, 18);

%% i) normalizd input
maxValue = max(x);
x_norm= x./ maxValue;

[TrainingError_norm,ValidError_norm]= KFoldCrossV(x_norm, y, 5, 18);
