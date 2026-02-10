cmake_minimum_required(VERSION 3.16)
project(ComputerVision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Put executables in build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ---------- OpenCV ----------
# Set this to the folder that contains OpenCVConfig.cmake
# For your MSVC vc16 OpenCV, that's usually:
#   <opencv_root>/build/x64/vc16/lib
#
# You can set it here OR pass it on the command line: -DOpenCV_DIR="..."
if(NOT OpenCV_DIR)
  set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/opencv/build/x64/vc16/lib" CACHE PATH "Path to OpenCVConfig.cmake folder")
endif()

find_package(OpenCV REQUIRED)

# Copy OpenCV DLLs next to the exe (so you don't need PATH).
# Adjust OPENCV_BIN_DIR if your OpenCV is elsewhere.
set(OPENCV_BIN_DIR "${CMAKE_SOURCE_DIR}/opencv/build/x64/vc16/bin")

function(copy_runtime_bits tgt)
  # Copy resources next to exe
  add_custom_command(TARGET ${tgt} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/resources"
            "$<TARGET_FILE_DIR:${tgt}>/resources"
  )

  # Copy OpenCV DLLs next to exe (only if directory exists)
  if(EXISTS "${OPENCV_BIN_DIR}")
    add_custom_command(TARGET ${tgt} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
              "${OPENCV_BIN_DIR}"
              "$<TARGET_FILE_DIR:${tgt}>"
    )
  endif()
endfunction()

# ---------- Apps: every app/*.cpp becomes its own .exe ----------
file(GLOB APP_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/app/*.cpp")

if(APP_SOURCES STREQUAL "")
  message(WARNING "No sources found in ${CMAKE_SOURCE_DIR}/app/*.cpp")
endif()

foreach(app_src ${APP_SOURCES})
  get_filename_component(app_name ${app_src} NAME_WE)
  add_executable(${app_name} ${app_src})
  target_link_libraries(${app_name} PRIVATE ${OpenCV_LIBS})
  target_include_directories(${app_name} PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_compile_definitions(${app_name} PRIVATE RESOURCES_DIR="${CMAKE_SOURCE_DIR}/resources")

  copy_runtime_bits(${app_name})
endforeach()
