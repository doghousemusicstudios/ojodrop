# Aggregator CMake project that builds ONLY the static libraries the milk
# converter shim links — hlsl2glslfork + glsl-optimizer (glsl_optimizer,
# glcpp-library, mesa) — straight from the vendored submodule sources.
#
# This deliberately does NOT build milkdrop-shader-converter's `.node` Node addon
# target (which needs the Node/cmake-js headers). build.rs drives this via the
# `cmake` crate, passing CONVERTER_DIR = the vendored submodule root.
#
# Mirrors the upstream milkdrop-shader-converter/CMakeLists.txt settings (C++20,
# PIC, `-Dlinux` on UNIX) so the libs match the cmake-js-built prebuilts.
cmake_minimum_required(VERSION 3.10)
project(milk_converter_libs CXX C)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (UNIX)
  add_definitions(-Dlinux)
endif ()

# The vendored glsl-optimizer / hlsl2glslfork ship flex/bison-generated lexers
# that use the `register` storage class, removed in C++17 and a *default error*
# under recent clang (the global `-w` suppresses warnings but not this). Disable
# that specific diagnostic for all sub-targets added below.
add_compile_options(-Wno-register -Wno-deprecated-declarations)

if (NOT DEFINED CONVERTER_DIR)
  message(FATAL_ERROR "CONVERTER_DIR must point at the milkdrop-shader-converter submodule root")
endif ()

add_subdirectory(${CONVERTER_DIR}/glsl-optimizer glsl-optimizer EXCLUDE_FROM_ALL)
add_subdirectory(${CONVERTER_DIR}/hlsl2glslfork hlsl2glslfork EXCLUDE_FROM_ALL)

# Single target the `cmake` crate builds; pulls in all four static libs.
add_custom_target(milk_converter_libs ALL
  DEPENDS hlsl2glsl glsl_optimizer glcpp-library mesa)
