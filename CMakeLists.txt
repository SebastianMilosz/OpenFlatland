cmake_minimum_required(VERSION 2.8.11)

project (LIFESIM)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(LIFESIM_AppSRC
    src/application.cpp
    src/annviewerwidget.cpp
    src/colorizecircleshape.cpp
    src/colorizerealnbr.cpp
    src/consolewidget.cpp
    src/constelement.cpp
    src/constelementline.cpp
    src/constelementsfactory.cpp
    src/entity.cpp
    src/entityfactory.cpp
    src/entityghost.cpp
    src/entityshell.cpp
    src/entityvision.cpp
    src/fontfactory.cpp
    src/guiwidgetslayer.cpp
    src/informationwidget.cpp
    src/performancelogger.cpp
    src/physicsbody.cpp
    src/propertyeditorwidget.cpp
    src/serializableneuronlayer.cpp
    src/serializableneuronlayercontainer.cpp
    src/world.cpp
)

# These are used to create visual studio folders.
source_group("src" FILES ${LIFESIM_AppSRC})

# These are used to create visual studio folders.
source_group("" FILES main.cpp)

include_directories( include )
include_directories( libraries/SFML-master/src )

add_definitions(-DSFML_STATIC)
add_definitions(-DGLEW_STATIC)
add_definitions(-DSERIALIZABLE_USE_LUA)

# Add sub-directories
add_subdirectory(libraries/codeframe-master libraries/codeframe-master )
add_subdirectory(libraries/Box2D-master/Box2D libraries/Box2D-master/Box2D )
add_subdirectory(libraries/SFML-master libraries/SFML-master )

add_executable(LIFESIM_App
    ${LIFESIM_AppSRC}
)

set_target_properties(LIFESIM_App PROPERTIES
    OUTPUT_NAME "LIFESIM"
    CLEAN_DIRECT_OUTPUT 1
)

target_include_directories( LIFESIM_App
    PUBLIC ${PROJECT_SOURCE_DIR}/include )
    
target_include_directories( LIFESIM_App
    PUBLIC ${PROJECT_SOURCE_DIR}/libraries/codeframe-master/include )

target_link_libraries(LIFESIM_App
    codeframe
)