cmake_minimum_required(VERSION 3.7.1)

project (LIFESIM)

# enable project folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMake")

set(LIFESIM_AppSRC
    main.cpp
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

set(LIFESIM_ImGuiSRC
    libraries/imgui-master/imgui.cpp
    libraries/imgui-master/imgui_demo.cpp
    libraries/imgui-master/imgui_draw.cpp
    libraries/imgui-master/imgui_widgets.cpp
    libraries/imgui-sfml-master/imgui-SFML.cpp
)

add_definitions(-DSFML_STATIC)
add_definitions(-DGLEW_STATIC)
add_definitions(-DSERIALIZABLE_USE_LUA)
add_definitions(-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)

include_directories( include )
include_directories( libraries/CTPL-master )
include_directories( libraries/TinyThread++-1.0/source )
include_directories( libraries/SFML-master/include )
include_directories( libraries/imgui-sfml-master )
include_directories( libraries/imgui-master )
include_directories( libraries/Box2D-master/Box2D )
include_directories( libraries/thrust-1.8.1 )

set(BOX2D_BUILD_EXAMPLES     ON  CACHE BOOL "BOX2D build examples")
set(BOX2D_BUILD_STATIC       ON  CACHE BOOL "BOX2D static build")
set(BOX2D_BUILD_SHARED       OFF CACHE BOOL "BOX2D shared build")
set(SFML_USE_STATIC_STD_LIBS ON  CACHE BOOL "SFML use static std libs")
set(SFML_BUILD_AUDIO         ON  CACHE BOOL "SFML build audio")
set(BUILD_SHARED_LIBS        OFF CACHE BOOL "Build shared libs")

# Add sub-directories
add_subdirectory(libraries/codeframe-master )
add_subdirectory(libraries/codeframe-master/thirdparty/lua-5.3.5/src )
add_subdirectory(libraries/codeframe-master/thirdparty/luabridge-master )
add_subdirectory(libraries/Box2D-master/Box2D )
add_subdirectory(libraries/SFML-master )

add_executable(LIFESIM_App
    ${LIFESIM_AppSRC}
    ${LIFESIM_ImGuiSRC}
)

set_target_properties(LIFESIM_App PROPERTIES
    OUTPUT_NAME "LIFESIM"
    CLEAN_DIRECT_OUTPUT 1
)

set_target_properties(LIFESIM_App PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_include_directories( LIFESIM_App
    PUBLIC ${PROJECT_SOURCE_DIR}/include )
    
target_include_directories( LIFESIM_App
    PUBLIC ${PROJECT_SOURCE_DIR}/libraries/codeframe-master/include )
   
target_link_libraries(LIFESIM_App
    codeframe sfml-main sfml-graphics sfml-network sfml-window sfml-system Box2D lualib
)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/arial.ttf
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
        