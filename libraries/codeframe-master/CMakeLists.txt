cmake_minimum_required(VERSION 2.8.11)

project (CODEFRAME_LIB)

SET(CMAKE_CXX_FLAGS "-std=c++0x")

set(CODEFRAME_LIB_HDRS

)

set(CODEFRAME_LIB_SRC
    src/base64.cpp
    src/cxml.cpp  
    src/cxmlnode.cpp
    src/extpoint2d.cpp
    src/extthrust.cpp
    src/extvector.cpp
    src/instancemanager.cpp
    src/referencemanager.cpp
    src/serializable.cpp
    src/serializablechildlist.cpp
    src/serializablecontainer.cpp
    src/serializableidentity.cpp
    src/serializableinterface.cpp
    src/serializablelua.cpp
    src/serializablepath.cpp
    src/serializableproperty.cpp
    src/serializablepropertyinfo.cpp
    src/serializablepropertyiterator.cpp
    src/serializablepropertymanager.cpp
    src/serializableregister.cpp
    src/serializableselectable.cpp
    src/serializablestorage.cpp
    src/serializablestream.cpp
    src/serializablestreammanager.cpp
    src/typeinfo.cpp
    src/xmlformatter.cpp
)

set(CODEFRAME_LIB_UTILITIES_SRC
    src/utilities/FileUtilities.cpp
    src/utilities/LoggerUtilities.cpp
    src/utilities/PerformanceUtilities.cpp
    src/utilities/SysUtilities.cpp
    src/utilities/ThreadUtilities.cpp
)

add_library(CODEFRAME_LIB STATIC
    ${CODEFRAME_LIB_HDRS}
    ${CODEFRAME_LIB_SRC}
    ${CODEFRAME_LIB_UTILITIES_SRC}
)

# Some external dependencies
include_directories( include )
include_directories( include/utilities )
include_directories( thirdparty/pugixml-1.7/src )
include_directories( thirdparty/sigslot-1.0.0 )
include_directories( thirdparty/smartpointer )
include_directories( ../SFML-master/include )
include_directories( ../thrust-1.8.1 )

add_library(codeframe ALIAS ${PROJECT_NAME})

add_definitions(-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)

# Add sub-directories
add_subdirectory(thirdparty)
add_subdirectory(tests)

option (USE_QT "Use QT" OFF)  

if (USE_QT)
set ( CMAKE_PREFIX_PATH "C:\\Qt\\Qt5.1.0\\Tools\\mingw48_32\\bin" )
set ( CMAKE_PREFIX_PATH "C:\\Qt\\Qt5.0.1\\5.0.1\\mingw48_32" )
add_subdirectory(qtexample)
endif (USE_QT)