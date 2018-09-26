function w = WeightedLinearRegress( x, y, U)
%LinearRegress is w = (vecx'*vecx)^-1 *vecx' * y;

vecx=[x, ones(length(x), 1)];
w = (vecx'*U*vecx)^-1* vecx'  * U * y;
