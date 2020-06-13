cmake_minimum_required(VERSION 3.5)
project(dwa_demo)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_C_COMPILER g++)
add_compile_options(-std=c++11)

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/modules")

find_package(OpenCV 3.4.0 REQUIRED)

find_package(CUDA)
set(CUDA_SOURCE_PROPERTY_FORMAT OBJ)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS -arch=sm_60;-O3;-G;-g;-std=c++11)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    )

add_executable(dwa_cpp_demo
cpp_sample/dwa_demo.cpp cpp_sample/dwa.cpp cpp_sample/dwa.h cpp_sample/utility.h)

target_link_libraries(dwa_cpp_demo
    ${OpenCV_LIBS}
)


include_directories(
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)

CUDA_ADD_EXECUTABLE(dwa_cuda_demo
        cuda_sample/dwa_demo.cpp
        cuda_sample/dwa.cu)

target_link_libraries(dwa_cuda_demo
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBS}
        )


