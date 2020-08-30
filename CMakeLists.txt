cmake_minimum_required(VERSION 3.8)

project (OpenFlatland VERSION 0.1.1)

# enable project folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMake")

set(OpenFlatland_AppSRC
    main.cpp
    src/application.cpp
    src/ann_viewer_widget.cpp
    src/entity_vision_viewer_widget.cpp
    src/colorize_circle_shape.cpp
    src/colorize_ray_data.cpp
    src/console_widget.cpp
    src/const_element.cpp
    src/const_element_line.cpp
    src/const_elements_factory.cpp
    src/entity.cpp
    src/entity_factory.cpp
    src/entity_ghost.cpp
    src/entity_shell.cpp
    src/entity_motion.cpp
    src/drawable_entity_energy.cpp
    src/drawable_entity_motion.cpp
    src/drawable_entity_vision.cpp
    src/drawable_object.cpp
    src/entity_energy.cpp
    src/entity_vision.cpp
    src/entity_vision_node.cpp
    src/ray_data.cpp
    src/font_factory.cpp
    src/gui_widgets_layer.cpp
    src/information_widget.cpp
    src/performance_logger.cpp
    src/physics_body.cpp
    src/property_editor_widget.cpp
    src/neuron_cell_pooll.cpp
    src/neuron_layer.cpp
    src/neuron_layer_container.cpp
    src/neuron_layer_ray.cpp
    src/neuron_layer_vector.cpp
    src/artificial_neuron_engine.cpp
    src/world.cpp
)

set(OpenFlatland_ImGuiSRC
    libraries/imgui/imgui.cpp
    libraries/imgui/imgui_demo.cpp
    libraries/imgui/imgui_draw.cpp
    libraries/imgui/imgui_widgets.cpp
    libraries/imgui-sfml/imgui-SFML.cpp
)

add_definitions(-DSFML_STATIC)
add_definitions(-DGLEW_STATIC)
add_definitions(-DSERIALIZABLE_USE_LUA)
add_definitions(-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)

include_directories( include )
include_directories( libraries/CTPL-master )
include_directories( libraries/TinyThread++-1.0/source )
include_directories( libraries/SFML-master/include )
include_directories( libraries/imgui-sfml )
include_directories( libraries/imgui )
include_directories( libraries/box2d/include )
include_directories( libraries/thrust )

set(BOX2D_BUILD_EXAMPLES     OFF CACHE BOOL "BOX2D build examples")
set(BOX2D_BUILD_STATIC       ON  CACHE BOOL "BOX2D static build")
set(BOX2D_BUILD_SHARED       OFF CACHE BOOL "BOX2D shared build")
set(BOX2D_BUILD_TESTBED      OFF CACHE BOOL "BOX2D testbed build")
set(BOX2D_BUILD_UNIT_TESTS   OFF CACHE BOOL "BOX2D unit tests build")
set(SFML_USE_STATIC_STD_LIBS ON  CACHE BOOL "SFML use static std libs")
set(SFML_BUILD_AUDIO         OFF CACHE BOOL "SFML build audio")
set(BUILD_SHARED_LIBS        OFF CACHE BOOL "Build shared libs")

# Add sub-directories
add_subdirectory(libraries/codeframe-master )
add_subdirectory(libraries/codeframe-master/thirdparty/lua-5.3.5/src )
add_subdirectory(libraries/codeframe-master/thirdparty/luabridge-master )
add_subdirectory(libraries/box2d )
add_subdirectory(libraries/SFML-master )

add_executable(OpenFlatland_App
    ${OpenFlatland_AppSRC}
    ${OpenFlatland_ImGuiSRC}
)

set_target_properties(OpenFlatland_App PROPERTIES
    OUTPUT_NAME "OpenFlatland"
    CLEAN_DIRECT_OUTPUT 1
)

set_target_properties(OpenFlatland_App PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_include_directories( OpenFlatland_App
    PUBLIC ${PROJECT_SOURCE_DIR}/include )
    
target_include_directories( OpenFlatland_App
    PUBLIC ${PROJECT_SOURCE_DIR}/libraries/codeframe-master/include )
   
target_link_libraries( OpenFlatland_App
    codeframe 
    sfml-graphics 
    sfml-network 
    sfml-window 
    sfml-system 
    box2d 
    lualib
    pthread
)

if(SFML_OS_WINDOWS OR SFML_OS_ANDROID OR SFML_OS_IOS)
target_link_libraries( OpenFlatland_App sfml-main )
endif()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/arial.ttf
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/examples/OpenFlatland_cfg.xml
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
        