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
find_package(Eigen3 REQUIRED)


include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    )

add_executable(dwa_demo 
cpp_sample/dwa.cpp)

target_link_libraries(dwa_demo
    ${OpenCV_LIBS}
)
