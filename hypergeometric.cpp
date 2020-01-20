#include <pybind11/pybind11.h>
#include <boost/math/distributions/hypergeometric.hpp>
#include <boost/python.hpp>

namespace py = pybind11;

double hyperGeometricFormula(int numAntigenAttachedToDC, int numAntigenInContactArea, int tCellActivationThreshold,
double A) {
    if(numAntigenAttachedToDC*A<tCellActivationThreshold) return 0.0;
    if(numAntigenInContactArea<tCellActivationThreshold) return 0.0;
    if(A==0) return 0.0;
    if(A==1) return 1.0;
    boost::math::hypergeometric_distribution <> hypg(A*numAntigenAttachedToDC, numAntigenInContactArea, numAntigenAttachedToDC);
    return 1.0-cdf(hypg, tCellActivationThreshold-1);
}

PYBIND11_MODULE(hypergeometric, m) {
m.doc() = "pybind11 plugin returning hypergeometric distribution";
m.def("hyperGeometricFormula", &hyperGeometricFormula, "function returning cdf of hypergeometric formula");
}
