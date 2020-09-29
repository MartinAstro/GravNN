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

target_link_libraries(${TARGET_NAME} ${PYTHON_LIBRARIES})

set_target_properties(${TARGET_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}") 
