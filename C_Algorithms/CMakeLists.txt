
cmake_minimum_required(VERSION 2.8)
if(POLICY CMP0078)
  cmake_policy(SET CMP0078 NEW)
endif()
if(POLICY CMP0086)
  cmake_policy(SET CMP0086 NEW)
endif()


project("C_Algorithms")

find_package(PythonLibs 3 REQUIRED)
find_package(SWIG REQUIRED)
include(UseSWIG REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)


set(CMAKE_INCLUDE_CURRENT_DIR TRUE)


add_subdirectory(PinesAlgorithm)
add_subdirectory(Regression)

