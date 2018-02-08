cmake_minimum_required(VERSION 2.8.11)

project (CODEFRAME_QTEXAMPLE)

set ( SOURCES
 main.cpp
 mainwindow.cpp
)

set ( MOC_HEADERS
 mainwindow.h
)

set ( UIS
 mainwindow.ui
)

set ( RESOURCES
 qtexample.qrc
)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)

# Next lines needed for building all Qt projects
find_package( Qt5Widgets )

# Tell CMake to create the executable
add_executable(CODEFRAME_QTEXAMPLE ${SOURCES})

# The Qt5Widgets_INCLUDES also includes the include directories for
  # dependencies QtCore and QtGui
  include_directories(${Qt5Widgets_INCLUDES})

  # We need add -DQT_WIDGETS_LIB when using QtWidgets in Qt 5.
  add_definitions(${Qt5Widgets_DEFINITIONS})

  # Executables fail to build with Qt 5 in the default configuration
  # without -fPIE. We add that here.
  set(CMAKE_CXX_FLAGS "${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")

# Use the Widgets module from Qt 5.
target_link_libraries(CODEFRAME_QTEXAMPLE Qt5::Widgets)