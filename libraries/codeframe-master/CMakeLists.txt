cmake_minimum_required(VERSION 2.8.11)

project (CODEFRAME_LIBSRC)

set(CODEFRAME_MAJOR_VERSION 0)
set(CODEFRAME_MINOR_VERSION 1)
set(CODEFRAME_PATCH_VERSION 0)
set(CODEFRAME_VERSION ${CODEFRAME_MAJOR_VERSION}.${CODEFRAME_MINOR_VERSION}.${CODEFRAME_PATCH_VERSION})
  
option (USE_QT "Use QT" OFF)  
  
# Offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")  

# Add sub-directories
add_subdirectory(thirdparty)
add_subdirectory(src)
add_subdirectory(tests)

if (USE_QT)
set ( CMAKE_PREFIX_PATH "C:\\Qt\\Qt5.1.0\\Tools\\mingw48_32\\bin" )
set ( CMAKE_PREFIX_PATH "C:\\Qt\\Qt5.0.1\\5.0.1\\mingw48_32" )
add_subdirectory(qtexample)
endif (USE_QT)