cmake_minimum_required(VERSION 3.16)
project(sim)



SET(CMAKE_BUILD_TYPE Release CACHE STRING
	"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
#	FORCE
	)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})


find_package(Eigen3 REQUIRED)

add_compile_options(-DHAVE_CSTDDEF)
include_directories(${DART_INCLUDE_DIRS})
include_directories(${TinyXML_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${Pybind11_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})


file(GLOB SOURCES "*.h" "*.cpp")
#list(FILTER SOURCES EXCLUDE REGEX "BVHToXML.cpp")
#list(FILTER SOURCES EXCLUDE REGEX "swing_cleanup.cpp")

add_library(sim ${SOURCES} )
target_link_libraries(sim ${DART_LIBRARIES} ${Pybind11_LIBRARIES} ${TinyXML_LIBRARIES} ${PYTHON_LIBRARIES})
