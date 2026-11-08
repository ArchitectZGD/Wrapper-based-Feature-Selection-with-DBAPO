% Levy flight (vectorized) step
%{
  input: number of dimensions
  output: 1*d vector; each element is a random step size
  gamma: factorial of all values except negative integers
  sigma: standard formula of the Mantegna algorithm
%}
function LF = levyFlightVec(d)
    beta = 1.5;
    sigma = (gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    u = randn(1,d) * sigma;
    v = randn(1,d);
    LF = u ./ abs(v).^(1/beta);
end