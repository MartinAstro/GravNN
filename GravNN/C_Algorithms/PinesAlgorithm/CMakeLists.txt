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

target_compile_options(${TARGET_NAME} PRIVATE  ${OpenMP_CXX_FLAGS})
target_include_directories(${TARGET_NAME} PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} ${OpenMP_CXX_LIBRARIES})

MESSAGE(${Python_LIBRARIES})
MESSAGE(${Python_INCLUDE_DIRS})
target_include_directories(${TARGET_NAME} PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} ${Python_LIBRARIES})

# SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DMS_WIN64")
# SET(CMAKE_C_FLAGS  "${CMAKE_CXX_FLAGS} -DMS_WIN64")

#target_compile_definitions(${TARGET_NAME} PUBLIC "-D MS_WIN64")
#set_target_properties(${TARGET_NAME} PROPERTIES DEBUG_POSTFIX  "_d")

set_target_properties(${TARGET_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}") 
set_target_properties(${TARGET_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}") 
set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}") 

set_target_properties(${TARGET_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}") 
set_target_properties(${TARGET_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}") 
set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}") 

set_target_properties(${TARGET_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}") 
set_target_properties(${TARGET_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}") 
set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}") 


