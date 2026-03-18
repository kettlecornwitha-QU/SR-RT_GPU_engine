cmake_minimum_required(VERSION 3.20)
project(SR_RT_GPU_engine LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/render_sphere.metal
  ${CMAKE_CURRENT_BINARY_DIR}/render_sphere.metal
  COPYONLY
)

add_executable(sr_rt_gpu
  src/main.mm
  src/scene_builder.cpp
  src/renderer_host.mm
)

target_include_directories(sr_rt_gpu PRIVATE include)

target_compile_options(sr_rt_gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-O2 -Wall -Wextra -pedantic>
  $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
)

target_link_libraries(sr_rt_gpu PRIVATE
  "-framework Foundation"
  "-framework Metal"
)
