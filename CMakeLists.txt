
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

add_compile_options(-D LOCAL_RENDER)

# set the directory of executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(OpenCV_DIR ${OPENCV_DIR})
set(Target ${TARGET})

# find required opencv
find_package(OpenCV REQUIRED)
# directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS})
# name of executable file and path of source file
add_executable(${Target} main/${Target}.cpp)
# directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})
# opencv libraries
target_link_libraries(${Target} ${OpenCV_LIBS})

# includes
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/ncnn)
# target_link_libraries (example ${PROJECT_SOURCE_DIR}/lib/libncnn.a -fopenmp)
target_link_libraries (${Target} ${NCNN_INSTALL_DIR}/lib/libncnn.a -fopenmp)

# .so link
# target_link_libraries (example ${PROJECT_SOURCE_DIR}/lib/libzbar.so)
target_link_libraries (${Target} ${ZBAR_INSTALL_DIR}/lib/libzbar.so)

FILE(GLOB_RECURSE PRJ_SRC
        "${PROJECT_SOURCE_DIR}/src/*.c"
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
        )

add_library(mainlib ${PRJ_SRC})

target_link_libraries (mainlib ${NCNN_INSTALL_DIR}/lib/libncnn.a -fopenmp)
# target_link_libraries (mainlib ${PROJECT_SOURCE_DIR}/lib/libncnn.a -fopenmp)

target_link_libraries(${Target} mainlib)