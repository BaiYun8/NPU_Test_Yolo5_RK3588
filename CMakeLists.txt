cmake_minimum_required(VERSION 3.5)
project(cv)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_RKNN_PROFILING "Enable RKNN profiling output" OFF)

message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

set(LIB_ARCH "aarch64")
set(DEVICE_NAME "RK3588")

set(3DRPARTY_PATH       ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
set(RKNN_API_PATH       ${3DRPARTY_PATH}/librknn_api)
set(RKNN_INCLUDE_DIRS   ${RKNN_API_PATH}/include)
set(RKNN_LIBS           ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)

set(RGA_PATH            ${3DRPARTY_PATH}/rga/${DEVICE_NAME})
set(RGA_INCLUDE_DIRS    ${RGA_PATH}/include)
set(RGA_LIBS            ${RGA_PATH}/lib/Linux/${LIB_ARCH}/librga.so)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_INCLUDE_DIRS}
    ${RGA_INCLUDE_DIRS}
)

add_executable(app
    main.cpp
    thread_poll.cpp
    yolov5s.cpp
    post_process.cpp
)

if(ENABLE_RKNN_PROFILING)
    target_compile_definitions(app PRIVATE ENABLE_RKNN_PROFILING=1)
endif()

target_link_libraries(app
    ${OpenCV_LIBS}
    ${RKNN_LIBS}
    ${RGA_LIBS}
)
