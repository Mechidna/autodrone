cmake_minimum_required(VERSION 3.16)
project(aigp_openvins_replay LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OPENVINS_ROOT "$ENV{HOME}/src/open_vins" CACHE PATH "OpenVINS source root")
if(NOT EXISTS "${OPENVINS_ROOT}/ov_msckf/src/core/VioManager.h")
  message(FATAL_ERROR "OPENVINS_ROOT is not an OpenVINS checkout: ${OPENVINS_ROOT}")
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Boost REQUIRED COMPONENTS date_time filesystem system)
find_package(Ceres REQUIRED)
find_package(Threads REQUIRED)

find_library(
  OPENVINS_LIBRARY
  NAMES ov_msckf_lib
  PATHS "${OPENVINS_ROOT}/ov_msckf/build"
  NO_DEFAULT_PATH
)
if(NOT OPENVINS_LIBRARY)
  message(FATAL_ERROR
    "libov_msckf_lib was not found under ${OPENVINS_ROOT}/ov_msckf/build; "
    "build OpenVINS ROS-free first"
  )
endif()

add_executable(run_vq1_dataset run_vq1_dataset.cpp)
target_compile_definitions(run_vq1_dataset PRIVATE ROS_AVAILABLE=0)
target_include_directories(run_vq1_dataset PRIVATE
  "${OPENVINS_ROOT}/ov_msckf/src"
  "${OPENVINS_ROOT}/ov_core/src"
  "${OPENVINS_ROOT}/ov_init/src"
  ${Boost_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)
target_link_libraries(run_vq1_dataset PRIVATE
  "${OPENVINS_LIBRARY}"
  Eigen3::Eigen
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
  ${CERES_LIBRARIES}
  Threads::Threads
)
get_filename_component(OPENVINS_LIBRARY_DIR "${OPENVINS_LIBRARY}" DIRECTORY)
set_target_properties(run_vq1_dataset PROPERTIES
  BUILD_RPATH "${OPENVINS_LIBRARY_DIR}"
  INSTALL_RPATH "${OPENVINS_LIBRARY_DIR}"
)
