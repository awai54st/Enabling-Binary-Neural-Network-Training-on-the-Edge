#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <cstdlib>
#include "../utils/data_type.h"
#include "../utils/pybind_data_type.h"

namespace py = pybind11;


Matrix<float> mat_sum(Matrix<float>& vs) {
    /*float res = 0;
    const int size = vs.size;
    for (int i=0; i<size; i++) {
        res += vs.data[i];
    }*/
    return vs;
}

double vsum(const std::vector<double>& vs) {
    double res = 0;
    for (const auto& i: vs) {
        res += i;
    }
    return res;
}

std::vector<float> range(int start, int stop, int step) {
    std::vector<float> res;
    for (int i=start; i<stop; i+=step) {
        res.push_back(float(i));
    }
    return res;
}


void mat_sum_test() {
    range(0, 1000000, 1);
    return ;
}
PYBIND11_MODULE(ex5, m) {
    m.def("vsum", &vsum);
    m.def("range", &range);
    m.def("mat_sum", &mat_sum);
    m.def("mat_sum_test", &mat_sum_test);
}


/*
class Matrix {
    public:
        size_t rows, cols;
        std::vector<float> data;
        Matrix(): rows(0), cols(0){};
        Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
        }
        void set_data(float *ptr) { 
            std::vector<float> vec_buf(ptr, ptr+(cols*rows));
            data = vec_buf;
        }
};

PYBIND11_MODULE(ex5, m) {
    m.def("vsum", &vsum);
    m.def("range", &range);
    m.def("mat_sum", &mat_sum);
    
    
    py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
        .def_buffer([](Matrix &m) -> py::buffer_info {
            return py::buffer_info(
                m.data.data(),                               // Pointer to buffer
                sizeof(float),                          // Size of one scalar
                py::format_descriptor<float>::format(), // Python struct-style format descriptor
                2,                                      // Number of dimensions
                { m.rows, m.cols },                 // Buffer dimensions
                { sizeof(float) * m.cols,             // Strides (in bytes) for each index
                  sizeof(float) }
            );
        })
        .def(py::init([](py::buffer b) {
            // Request a buffer descriptor from Python
            py::buffer_info info = b.request();

            // Some sanity checks ...
            if (info.format != py::format_descriptor<float>::format())
                throw std::runtime_error("Incompatible format: expected a float array!");

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible buffer dimension!");
            Matrix m;
            //size_t m_rows = info.strides[rowMajor ? 0 : 1] / (py::ssize_t)sizeof(Scalar);
            m.rows = info.shape[0];
            //size_t m_cols = info.strides[rowMajor ? 1 : 0] / (py::ssize_t)sizeof(Scalar);
            m.cols = info.shape[1];
            m.set_data(static_cast<float *>(info.ptr));
            return m;
        }));
}
*/
