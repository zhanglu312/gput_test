cmake_minimum_required(VERSION 3.6)
project(gputest)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
find_package(OpenCV 3.3.0 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})
include_directories(include)

add_executable(TEST test/test.cpp)
target_link_libraries(TEST ${OpenCV_LIBS})

#add_library(faultCode SHARED ${SRC_REG})

