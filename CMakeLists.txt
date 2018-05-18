project(xacc-vqe-bayesopt CXX)
cmake_minimum_required(VERSION 3.2)
set(CMAKE_DISABLE_IN_SOURCE_BUILDS ON)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

option(VQEBAYESOPT_BUILD_TESTS "Build test programs" OFF)

find_package(XACC REQUIRED)

set(XACCVQE_DIR ${XACC_DIR})
find_package(XACCVQE REQUIRED)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${XACC_ROOT}" CACHE PATH "default install path" FORCE )
endif()

set(CppMicroServicesConfig_DIR "${XACC_ROOT}/share/cppmicroservices4/cmake")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${XACC_ROOT}/share/xacc")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CppMicroServicesConfig_DIR}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

include(tests)
include(CppMicroServicesConfig)

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

include_directories(${XACC_INCLUDE_DIRS})
include_directories(${XACCVQE_INCLUDE_DIRS})

link_directories(${XACC_LIBRARY_DIR})
link_directories(${XACC_ROOT}/plugins)
set(BOOST_ROOT ${XACC_ROOT})
set(BAYESOPT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BAYESOPT_BUILD_SHARED ON CACHE BOOL "" FORCE)
add_subdirectory(tpls/bayesopt)
add_subdirectory(backend)

