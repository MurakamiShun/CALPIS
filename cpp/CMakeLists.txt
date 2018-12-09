cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
#set(CMAKE_CXX_COMPILER "clang++")

find_package(Torch REQUIRED)
find_package(Siv3D REQUIRED)

add_library(calpis STATIC calpis.cpp)
target_link_libraries(calpis "${TORCH_LIBRARIES}")

add_executable(app main.cpp)
set_target_properties(app PROPERTIES CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

target_link_libraries(app "${SIV3D_LIBRARIES}" calpis "${TORCH_LIBRARIES}")


message(STATUS "${SIV3D_BUILD_TYPE}")
