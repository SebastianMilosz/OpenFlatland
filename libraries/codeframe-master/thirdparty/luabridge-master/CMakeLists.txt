cmake_minimum_required (VERSION 2.8.11)

project (LuaBridge)

set (CMAKE_CXX_STANDARD 11)

option (LUABRIDGE_NO_CXX11 "Use C++11 standard if supported by compiler" OFF)

if (LUABRIDGE_NO_CXX11)
  add_definitions (-DLUABRIDGE_NO_CXX11)
endif ()

add_subdirectory (Source)
