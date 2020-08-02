cmake_minimum_required(VERSION 3.8)

project (CODEFRAME_LIB VERSION 1.0.0)

set(CODEFRAME_LIB_HDRS
    include/base64.hpp
    include/cxml.hpp
    include/cxmlnode.hpp
    include/extpoint2d.hpp
    include/extthrust.hpp
    include/extvector.hpp
    include/instance_manager.hpp
    include/reference_manager.hpp
    include/serializable_object.hpp
    include/serializable_object_list.hpp
    include/serializable_object_container.hpp
    include/serializable_object_selection.hpp
    include/serializable_object_multiple_selection.hpp
    include/serializable_identity.hpp
    include/serializable_object_node.hpp
    include/serializable_property_base.hpp
    include/serializable_lua.hpp
    include/serializable_path.hpp
    include/serializable_property.hpp
    include/serializable_property_info.hpp
    include/serializable_property_iterator.hpp
    include/serializable_property_list.hpp
    include/serializable_property_selection.hpp
    include/serializable_property_multiple_selection.hpp
    include/serializable_property_node.hpp
    include/serializable_register.hpp
    include/serializable_selectable.hpp
    include/serializable_storage.hpp
    include/typeinfo.hpp
    include/typedefs.hpp
    include/xmlformatter.hpp
)

set(CODEFRAME_LIB_SRC
    src/base64.cpp
    src/cxml.cpp  
    src/cxmlnode.cpp
    src/extpoint2d.cpp
    src/extthrust.cpp
    src/extvector.cpp
    src/instance_manager.cpp
    src/reference_manager.cpp
    src/serializable_object.cpp
    src/serializable_object_list.cpp
    src/serializable_object_container.cpp
    src/serializable_object_selection.cpp
    src/serializable_object_multiple_selection.cpp
    src/serializable_identity.cpp
    src/serializable_object_node.cpp
    src/serializable_property_base.cpp
    src/serializable_lua.cpp
    src/serializable_path.cpp
    src/serializable_property.cpp
    src/serializable_property_info.cpp
    src/serializable_property_iterator.cpp
    src/serializable_property_list.cpp
    src/serializable_property_selection.cpp
    src/serializable_property_multiple_selection.cpp
    src/serializable_register.cpp
    src/serializable_selectable.cpp
    src/serializable_storage.cpp
    src/typeinfo.cpp
    src/xmlformatter.cpp
)

set(CODEFRAME_LIB_UTILITIES_SRC
    src/utilities/FileUtilities.cpp
    src/utilities/LoggerUtilities.cpp
    src/utilities/PerformanceUtilities.cpp
    src/utilities/SysUtilities.cpp
    src/utilities/MathUtilities.cpp
    src/utilities/ThreadUtilities.cpp
)

set(CODEFRAME_THIRD_PARTY_HDRS
    thirdparty/pugixml-1.7/src/pugiconfig.hpp
    thirdparty/pugixml-1.7/src/pugixml.hpp
    thirdparty/sigslot-1.0.0/sigslot.h
    thirdparty/smartpointer/smartpointer.h
)

set(CODEFRAME_THIRD_PARTY_SRC
    thirdparty/pugixml-1.7/src/pugixml.cpp
)

add_library(CODEFRAME_LIB STATIC
    ${CODEFRAME_LIB_HDRS}
    ${CODEFRAME_LIB_SRC}
    ${CODEFRAME_LIB_UTILITIES_SRC}
    ${CODEFRAME_THIRD_PARTY_HDRS}
    ${CODEFRAME_THIRD_PARTY_SRC}
)

set_target_properties(CODEFRAME_LIB PROPERTIES
    OUTPUT_NAME "CODEFRAME"
    CLEAN_DIRECT_OUTPUT 1
)

set_target_properties(CODEFRAME_LIB PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

# Some external dependencies
include_directories( include )
include_directories( include/utilities )
include_directories( thirdparty/pugixml-1.7/src )
include_directories( thirdparty/sigslot-1.0.0 )
include_directories( thirdparty/smartpointer )
include_directories( thirdparty/plf_nanotimer-master )
include_directories( thirdparty/lua-5.3.5/src )
include_directories( thirdparty/luabridge-master/Source )

# SFML and thrust related datatypes
include_directories( ../SFML-master/include )
include_directories( ../thrust-1.8.1 )

target_include_directories( CODEFRAME_LIB PUBLIC include/utilities )
target_include_directories( CODEFRAME_LIB PUBLIC thirdparty/plf_nanotimer-master )
target_include_directories( CODEFRAME_LIB PUBLIC thirdparty/lua-5.3.5/src )
target_include_directories( CODEFRAME_LIB PUBLIC thirdparty/smartpointer )
target_include_directories( CODEFRAME_LIB PUBLIC thirdparty/sigslot-1.0.0 )

add_definitions(-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)

option (USE_QT "Use QT" OFF)  

if (USE_QT)
set ( CMAKE_PREFIX_PATH "C:\\Qt\\Qt5.1.0\\Tools\\mingw48_32\\bin" )
set ( CMAKE_PREFIX_PATH "C:\\Qt\\Qt5.0.1\\5.0.1\\mingw48_32" )
add_subdirectory(qtexample)
endif (USE_QT)

add_library(codeframe ALIAS ${PROJECT_NAME})

add_subdirectory(tests)
