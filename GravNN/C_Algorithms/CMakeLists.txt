
cmake_minimum_required(VERSION 3.8)
if(POLICY CMP0078)
  cmake_policy(SET CMP0078 NEW)
endif()
if(POLICY CMP0086)
  cmake_policy(SET CMP0086 NEW)
endif()

if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW) # Error on missing dependencies 
endif()

project("C_Algorithms")

find_package (Python COMPONENTS Interpreter Development)
find_package(SWIG REQUIRED)
include(UseSWIG REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)


MESSAGE(${Python_LIBRARIES})

# if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register -std=c++11")
# endif()


# if(MSVC)
#     MESSAGE("WOWOWO IN IF")
#     add_definitions(/MP)
#     add_definitions(/D _CRT_SECURE_NO_WARNINGS)
#     add_definitions(/D _WINSOCK_DEPRECATED_NO_WARNINGS)
#     add_definitions(/D _WIN32_WINNT=0x0501) # Targets Windows xp
#     add_definitions(/W3)
#     add_definitions(/wd"4913")
#     add_definitions(/wd"4251")
#     # Make sure we are using Multi-Threaded run time library
#     foreach(flag
#         CMAKE_C_FLAGS
#         CMAKE_C_FLAGS_DEBUG
#         CMAKE_C_FLAGS_RELEASE
#         CMAKE_CXX_FLAGS
#         CMAKE_CXX_FLAGS_DEBUG
#         CMAKE_CXX_FLAGS_RELEASE)
#         MESSAGE("BEFORE: ${${flag}}")
#         string(REPLACE "/D_DEBUG" "" "${flag}" "${${flag}}")
#         string(REPLACE "/MD" "/MT" "${flag}" "${${flag}}")
#         string(REPLACE "/MDd" "/MTd" "${flag}" "${${flag}}")
#         string(REPLACE "/MTd" "/MTd" "${flag}" "${${flag}}")
#         set("${flag}" "${${flag}} /EHsc")
#         MESSAGE("AFTER: ${${flag}}")
#     endforeach()
# endif()

set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS YES)
endif()

add_subdirectory(PinesAlgorithm)

