# set(OPENMP_LIBRARIES "/usr/local/Cellar/llvm/5.0.1/lib")
# set(OPENMP_INCLUDES "/usr/local/Cellar/llvm/5.0.1/include")

# Find OpenMP
# if(APPLE AND USE_OpenMP)
#     if(CMAKE_C_COMPILER_ID MATCHES "Clang")
#         set(OpenMP_C "${CMAKE_C_COMPILER}")
#         set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
#         set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
#         set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
#         set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
#         set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
#     endif()
#     if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#       set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
#       set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
#       set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
#       set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
#       set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
#       set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
#     endif()
# endif()

set(TARGET_NAME "PinesAlgorithm")

set(files
"PinesAlgorithm.cpp"
"PinesAlgorithm.h"
"PinesAlgorithm.i"
)

set_property(SOURCE "PinesAlgorithm.i" PROPERTY CPLUSPLUS ON)
set_property(SOURCE "PinesAlgorithm.i"  PROPERTY USE_TARGET_INCLUDE_DIRECTORIES TRUE) # Allows for include paths in the .i files

swig_add_library(${TARGET_NAME} 
                                LANGUAGE "python"
                                TYPE MODULE
                                #OUTFILE_DIR "${CMAKE_BINARY_DIR}/Basilisk/${PARENT_DIR}" #_wrap.c/.cxx file
                                #OUTPUT_DIR "${CMAKE_BINARY_DIR}/Basilisk/${PARENT_DIR}") # generated .py file
                                OUTFILE_DIR "${CMAKE_CURRENT_BINARY_DIR}" #_wrap.c/.cxx file
                                OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}" # generated .py file
                                SOURCES ${files})

target_include_directories(${TARGET_NAME} PRIVATE ${PYTHON_INCLUDE_PATH})

target_compile_options(${TARGET_NAME} PRIVATE  ${OpenMP_CXX_FLAGS})
target_include_directories(${TARGET_NAME} PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} ${OpenMP_CXX_LIBRARIES})
target_link_libraries(${TARGET_NAME} ${PYTHON_LIBRARIES})

set_target_properties(${TARGET_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}") 
