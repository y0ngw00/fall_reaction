cmake_minimum_required(VERSION 3.16)
project(Yongwoo)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
		RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
# set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
# set (CMAKE_EXE_LINKER_FLAGS:STRING=-fsanitize=address  -fsanitize=leak)
# set (CMAKE_MODULE_LINKER_FLAGS:STRING=-fsanitize=address  -fsanitize=leak)

add_compile_options(-fPIC)
add_compile_options(-std=gnu++17)
add_compile_options(-Wdeprecated-declarations)
add_compile_options(-lm)

add_definitions(-DPROJECT_DIR="${CMAKE_HOME_DIRECTORY}")


set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake_module")
find_package(DART REQUIRED COMPONENTS gui optimizer-ipopt collision-bullet collision-ode CONFIG)
find_package(PythonLibs 3.6 REQUIRED)
find_package(PythonInterp 3.6 REQUIRED)
find_package(TinyXML REQUIRED)
find_package(pybind11 REQUIRED)


add_subdirectory( sim )
add_subdirectory( network )
#add_subdirectory( render )
add_subdirectory( render_ui )
add_subdirectory( imgui )



