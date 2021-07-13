# set compiler
set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)
set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
# project name
PROJECT(example)
# requirement of cmake version
cmake_minimum_required(VERSION 3.5)

# set the directory of executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(OpenCV_DIR ${OPENCV_DIR})
set(Target ${TARGET})

# find required opencv
find_package(OpenCV REQUIRED)
# directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS})
# name of executable file and path of source file
add_executable(example main/${Target}.cpp)
# directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})
# opencv libraries
target_link_libraries(example ${OpenCV_LIBS})

# includes
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/ncnn)
target_link_libraries (example ${PROJECT_SOURCE_DIR}/lib/libncnn.a -fopenmp)

# .so link
target_link_libraries (example ${PROJECT_SOURCE_DIR}/lib/libzbar.so)

FILE(GLOB_RECURSE PRJ_SRC
        "${PROJECT_SOURCE_DIR}/src/*.c"
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
        )

add_library(mainlib ${PRJ_SRC})

target_link_libraries (mainlib ${PROJECT_SOURCE_DIR}/lib/libncnn.a -fopenmp)

target_link_libraries(example mainlib)