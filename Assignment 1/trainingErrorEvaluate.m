function [ E ] = trainingErrorEvaluate( x, y, w )
%  cost function , hw(x)=x*w
E = 0.5*sum((x*w - y).^2);

end

