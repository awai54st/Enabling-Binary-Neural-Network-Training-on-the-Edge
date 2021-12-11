%%file ex5.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
namespace py = pybind11;

double vsum(const std::vector<double>& vs) {
    double res = 0;
    for (const auto& i: vs) {
        res += i;
    }
    return res;
}

std::vector<int> range(int start, int stop, int step) {
    std::vector<int> res;
    for (int i=start; i<stop; i+=step) {
        res.push_back(i);
    }
    return res;
}


PYBIND11_MODULE(ex5, m) {
    m.def("vsum", &vsum);
    m.def("range", &range);
}