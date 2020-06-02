%module PinesAlgorithm
%{
#include "PinesAlgorithm.h"
%}
%include "std_vector.i"

namespace std {
    %template(IntVector) vector<int>;
    %template(DoubleVector) vector<double>;
    %template(MultiArray) vector<vector<double> >;
}

%include "PinesAlgorithm.h"