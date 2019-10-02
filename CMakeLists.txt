cmake_minimum_required(VERSION 2.8.11)

project (LIFESIM)

set(LIFESIM_MAJOR_VERSION 0)
set(LIFESIM_MINOR_VERSION 1)
set(LIFESIM_PATCH_VERSION 0)
set(LIFESIM_VERSION ${LIFESIM_MAJOR_VERSION}.${LIFESIM_MINOR_VERSION}.${LIFESIM_PATCH_VERSION})
  
# Offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")  

# Add sub-directories
add_subdirectory(libraries\\Box2D-master\\Box2D)
add_subdirectory(libraries\\codeframe-master)
add_subdirectory(libraries\\SFML-master)