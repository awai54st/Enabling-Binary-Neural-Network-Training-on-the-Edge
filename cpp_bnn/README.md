# BNN
## C++ BNN Training Framework
To run the c++ application, please ensure that the environment is set up as follow: 
1. gcc version 9.3.0 (Ubuntu 9.3.0-17ubuntu1~20.04)

The C++ implementations are located in `core/cython/lib`
The C++ makefiles are located in `core/cython/build/cpp_profile`, in particular, these makefiles are the main makefiles used to build the C++ codes:
1. makefile_optimized_layers
2. makefile_naive_layers
3. makefile_integration_test

For small models, valgrind can be used directly to measure the memory usage.
The memory profiling technique for large models are: /usr/bin/time -v `run_file_name`
