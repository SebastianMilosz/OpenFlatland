#include "neuron_layer.hpp"

#include <utilities/MathUtilities.h>
#include <utilities/LoggerUtilities.h>

#include <iostream>
#include <random>
#include <chrono>

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
NeuronLayer::NeuronLayer( const std::string& name, ObjectNode* parent, const std::string& link ) :
    Object( name, parent )
{
}
