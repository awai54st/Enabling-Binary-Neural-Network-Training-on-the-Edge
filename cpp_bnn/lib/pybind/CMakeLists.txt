cmake_minimum_required(VERSION 3.16.3)

# set the project name
project(test_cmake VERSION 1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
SET(CMAKE_CXX_FLAGS "-O3 -ftree-vectorize -unroll-loops -mcpu=native -mtune=cortex-a53 -march=armv8-a+crc+simd+fp -mlittle-endian")
SET(CMAKE_EXE_LINKER_FLAGS  "-larmadillo")
SET(USE_ARMA_VERSION "9.800.4")

option(USE_EIGEN3 "Use Eigen3" OFF)
if(USE_EIGEN3)
    ## add_subdirectory(/usr/include/eigen3/)
    list(APPEND EXTRA_INCLUDES "/usr/include/eigen3/")
endif()

# pybind
SET(pybind11_DIR "/home/ubuntu/imperial_project/ml_cpu/lib/python3.8/site-packages/pybind11/share/cmake/pybind11/")
##find_package(Python COMPONENTS Interpreter Development NumPy)
## find_package(pybind11 CONFIG)
list(APPEND EXTRA_INCLUDES "/usr/include/python3.8")
list(APPEND EXTRA_INCLUDES "/home/ubuntu/imperial_project/ml_cpu/lib/python3.8/site-packages/pybind11/include/") # this works
list(APPEND EXTRA_INCLUDES "/home/ubuntu/imperial_project/ml_cpu/lib/python3.8/site-packages/numpy/core/include/")
list(APPEND EXTRA_INCLUDES "/home/ubuntu/imperial_project/BNN/core/cython/lib/pybind/")
## list(APPEND EXTRA_INCLUDES "/home/ubuntu/imperial_project/BNN/core/cython/lib/pybind/carma/carma_bits/")

# armadillo library
## find_library(ARMADILLO_LIB armadillo HINTS /usr/lib/ REQUIRED) # this works

# add the executable
add_executable(test_cmake pybind_main.cpp)
## target_link_libraries(test_cmake PUBLIC ${ARMADILLO_LIB})
## SET(armadillo_DIR "/usr/lib/")

target_link_libraries(
    test_cmake PUBLIC 
    ${EXTRA_LIBS}
    ${CMAKE_EXE_LINKER_FLAGS}
    #${ARMADILLO_LIB} # this works
)

target_include_directories(
    test_cmake PUBLIC
    #"${PYTHON_LIBRARY_DIR}"
    #"${Python_INCLUDE_DIRS}"
    #"${Python_NumPy_INCLUDE_DIRS}"
    "${PROJECT_BINARY_DIR}"
    "${EXTRA_INCLUDES}"
)
## set(CMAKE_CXX_STANDARD 17)
## set(CMAKE_CXX_STANDARD_REQUIRED ON)
## set(CMAKE_CXX_EXTENSIONS OFF)

## if(NOT CMAKE_BUILD_TYPE)
##   set(CMAKE_BUILD_TYPE Release)
## endif()

## set(CMAKE_CXX_FLAGS "-O3")
## set(CMAKE_CXX_FLAGS_RELEASE "-O3")


## include_directories("/usr/include")
## include_directories("${CMAKE_SOURCE_DIR}") # current dir
## include_directories("/usr/include/python3.8")
## include_directories("/home/ubuntu/carma")

## file (GLOB SOURCE_FILES "cpp/src/*.cpp")
## file (GLOB HEADER_FILES "cpp/include/simple_gibbs_bits/*.hpp")
## file (GLOB PYTHON_FILES "/home/ubuntu/carma/include/*.h" "/home/ubuntu/carma/include/carma_bits/*.h")

# Set up such that XCode organizes the files
## source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES} ${HEADER_FILES} ${PYTHON_FILES} )

# Armadillo
## find_library(ARMADILLO_LIB armadillo HINTS /usr/lib/ REQUIRED)

# Pybind11
## find_package(pybind11 HINTS /home/ubuntu/pybind11/build/ REQUIRED)
## pybind11_add_module(test_pybind_carma 
## 	${SOURCE_FILES}
## 	${HEADER_FILES}
## 	${PYTHON_FILES}
## )

## target_link_libraries(test_pybind_carma PUBLIC ${ARMADILLO_LIB})

## install(TARGETS test_pybind_carma
##   COMPONENT python
##   LIBRARY DESTINATION "${PYTHON_LIBRARY_DIR}"
##   )