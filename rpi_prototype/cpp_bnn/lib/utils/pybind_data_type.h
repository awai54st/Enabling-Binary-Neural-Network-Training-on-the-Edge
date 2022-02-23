#ifndef pybind_data_type_h
#define pybind_data_type_h

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "data_type.h"

// https://github.com/tdegeus/pybind11_examples/blob/master/09_numpy_cpp-custom-matrix/pybind_matrix.h
// https://alexsm.com/pybind11-buffer-protocol-opencv-to-numpy/
// https://github.com/pybind/pybind11/blob/master/include/pybind11/cast.h
namespace py = pybind11;

// type caster: Matrix <-> NumPy-array
namespace pybind11 { namespace detail {
    template <typename T> struct type_caster<Matrix<T>> {
        public:
            PYBIND11_TYPE_CASTER(Matrix<T>, _("Matrix<T>"));

            // Conversion part 1 (Python -> C++)
            bool load(py::handle src, bool convert) {
                if (!convert && !py::array_t<T>::check_(src))
                    return false;

                auto buf = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(src);
                if (!buf)
                    return false;

                auto dims = buf.ndim();
                if (dims < 1 || dims > 4)
                    return false;

                std::vector<size_t> shape(buf.ndim());

                for ( int i=0 ; i<buf.ndim() ; i++ )
                    shape[i] = buf.shape()[i];

                value = Matrix<T>(shape,buf.data());

                return true;
            }

        //Conversion part 2 (C++ -> Python)
        static py::handle cast(
            const Matrix<T>& src, py::return_value_policy policy, py::handle parent) {
            py::array a(std::move(src.shape()), std::move(src.strides(true)), src.data() );
            
            return a.release();
        }
    };
}}

/*
// wrap C++ function with NumPy array IO
py::array py_length(py::array_t<double, py::array::c_style | py::array::forcecast> array)
{
  // check input dimensions
  if ( array.ndim()     != 2 )
    throw std::runtime_error("Input should be 2-D NumPy array");
  if ( array.shape()[1] != 2 )
    throw std::runtime_error("Input should have size [N,2]");

  // allocate std::vector (to pass to the C++ function)
  std::vector<double> pos(array.size());

  // copy py::array -> std::vector
  std::memcpy(pos.data(),array.data(),array.size()*sizeof(double));

  // call pure C++ function
  std::vector<double> result = length(pos);

  ssize_t              ndim    = 2;
  std::vector<ssize_t> shape   = { array.shape()[0] , 3 };
  std::vector<ssize_t> strides = { sizeof(double)*3 , sizeof(double) };

  // return 2-D NumPy array
  return py::array(py::buffer_info(
    result.data(),                           // data as contiguous array
    sizeof(double),                          // size of one scalar      
    py::format_descriptor<double>::format(), // data type               
    ndim,                                    // number of dimensions    
    shape,                                   // shape of the matrix     
    strides                                  // strides for each axis   
  ));
}
*/
#endif