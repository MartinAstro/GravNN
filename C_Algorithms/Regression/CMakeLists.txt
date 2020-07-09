set(TARGET_NAME "Regression")
#set (UseSWIG_MODULE_VERSION 2)

set(files
 "Regression.cpp"
 "Regression.h"
 "Regression.i"
 "../PinesAlgorithm/PinesAlgorithm.cpp"
 "../PinesAlgorithm/PinesAlgorithm.h"
)

set_property(SOURCE "Regression.i" PROPERTY CPLUSPLUS ON)
set_property(SOURCE "Regression.i"  PROPERTY USE_TARGET_INCLUDE_DIRECTORIES TRUE) # Allows for include paths in the .i files

swig_add_library(${TARGET_NAME} 
                                LANGUAGE "python"
                                TYPE MODULE
                                SOURCES ${files})

target_include_directories(${TARGET_NAME} PRIVATE ${PYTHON_INCLUDE_PATH})

target_link_libraries(${TARGET_NAME} ${PYTHON_LIBRARIES})
target_link_libraries(${TARGET_NAME} Eigen3::Eigen)

