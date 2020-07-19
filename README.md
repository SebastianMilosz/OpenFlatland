# OpenFlatland

## Basic information
-----
This project is aim to create a sandbox for reinforcement learning experiments on artificial creatures that will act in complex simulated environment.

![screenshot](https://raw.githubusercontent.com/SebastianMilosz/OpenFlatland/master/doc/OpenFlatland_scr001.png)

-----
Basic class relations can be sawn on below diagram:
![BasicDependencyDiagram](https://raw.githubusercontent.com/SebastianMilosz/OpenFlatland/master/doc/BasicDependencyDiagram.jpg)

Basic entity class hierarchy:
* PhysicsBody - This class reprezent physicals of the entity, so it is connected with physical engine (Box2D library) that calculate interactions between objects.
* EntityShell - This class contains all sub components that will be used as inputs/outputs for neuron engine like: Vision, Motion, Energy ect..
* EntityGhost - 
* ArtificialNeuronEngine - 

Additional supporting classes:
* EntityVision - This class is responsible for getting environment visual informations using raycast. It create vector of distance data within some defined angle.
* EntityMotion - This class handle output data from NeuronEngine and translate those to linear and radius velocity it also calculate energy cost of actions.
* EntityEnergy - This class handle energy calculations and energy gathering from environment
* NeuronLayer - This is only interface data layer between entity and environment, we can add any number of inputs and outputs to the neuron engine

This project use codeframe library that provide additional meta layer to the code. 
We are able to control all objects within  the application
by its properties that are available automaticaly from gui interface:

![CodeFrameGuiInterface](https://raw.githubusercontent.com/SebastianMilosz/OpenFlatland/master/doc/CodeFrameGuiInterface.png)

## Dependencies
-----

* [CodeFrame](https://github.com/SebastianMilosz/OpenFlatland/master/libraries/codeframe-master)
* [ImGui+docking](https://github.com/ocornut/imgui)
* [ImGui + SFML](https://github.com/eliasdaler/imgui-sfml)
* [box2d](https://github.com/erincatto/box2d)
* [thrust](https://github.com/thrust/thrust)
* [lua](https://github.com/lua)

## Building
-----

* cmake --version
* mkdir build
* cd build
* cmake ..
* make