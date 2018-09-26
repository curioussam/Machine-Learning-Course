function [ X,w ] = PolyRegress( x, y, d )

X = [x, ones(length(x), 1)];

% adds x2...xd to the x
for power = 2:d
    X = [ x.^power, X];
end

% polynomial regression fomula
w = (X'*X)^-1*X' * y;

end

