cmake_minimum_required(VERSION 3.16)
project(aigp_orbslam3_vio_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED ORB_SLAM3_ROOT)
  if(DEFINED ENV{ORB_SLAM3_ROOT})
    set(ORB_SLAM3_ROOT "$ENV{ORB_SLAM3_ROOT}")
  else()
    message(FATAL_ERROR "Set ORB_SLAM3_ROOT to the ORB-SLAM3 checkout/build directory.")
  endif()
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin QUIET)
find_package(Boost REQUIRED COMPONENTS serialization)

set(ORB_SLAM3_LIBRARY "${ORB_SLAM3_ROOT}/lib/libORB_SLAM3.so")
if(NOT EXISTS "${ORB_SLAM3_LIBRARY}")
  message(FATAL_ERROR "Expected ORB-SLAM3 library at ${ORB_SLAM3_LIBRARY}. Build ORB-SLAM3 first.")
endif()

add_executable(
  orbslam3_vio_server
  "${CMAKE_CURRENT_LIST_DIR}/../orbslam3_vio_server.cpp"
)

target_include_directories(
  orbslam3_vio_server
  PRIVATE
    "${ORB_SLAM3_ROOT}"
    "${ORB_SLAM3_ROOT}/include"
    "${ORB_SLAM3_ROOT}/include/CameraModels"
    "${ORB_SLAM3_ROOT}/Thirdparty/Sophus"
    ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(
  orbslam3_vio_server
  PRIVATE
    "${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib"
    "${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib"
)

target_link_libraries(
  orbslam3_vio_server
  PRIVATE
    "${ORB_SLAM3_LIBRARY}"
    ${OpenCV_LIBS}
    Eigen3::Eigen
    Boost::serialization
)

if(Pangolin_FOUND)
  target_link_libraries(orbslam3_vio_server PRIVATE ${Pangolin_LIBRARIES})
  target_include_directories(orbslam3_vio_server PRIVATE ${Pangolin_INCLUDE_DIRS})
endif()

target_compile_options(
  orbslam3_vio_server
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
