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
SerializableNeuronLayer::SerializableNeuronLayer( const std::string& name, ObjectNode* parent ) :
    Object( name, parent ),
    Activation      ( this, "Activation", 0,
                        cPropertyInfo().
                            Kind(KIND_ENUM).
                            Enum("Identity,Binary step,Logistic").
                            Description("Activation Function")),
    WeightDimensions( this, "WeightDimensions", std::vector<unsigned int>(0),
                        cPropertyInfo().
                            Kind(KIND_VECTOR, KIND_NUMBER).
                            Description("WeightDimensions"),
                        std::bind(&SerializableNeuronLayer::GetWeightDimensionsConstVector, this),
                        nullptr,
                        std::bind(&SerializableNeuronLayer::GetWeightDimensionsVector, this)),
    WeightMatrix    ( this, "WeightMatrix", thrust::host_vector<float>(0),
                        cPropertyInfo().
                            Kind(KIND_VECTOR_THRUST_HOST, KIND_REAL).
                            Description("WeightMatrix")),
    Input           ( this, "Input", thrust::host_vector<RayData>(),
                        cPropertyInfo().
                            Kind(KIND_VECTOR_THRUST_HOST).
                            Description("Input")),
    Output          ( this, "Output", thrust::host_vector<float>(0),
                        cPropertyInfo().
                            Kind( KIND_VECTOR_THRUST_HOST, KIND_REAL ).
                            Description("Output"))
{
    // Signal On property change connection
    WeightDimensions.signalChanged.connect( this, &SerializableNeuronLayer::OnWeightDimensionsVectorChanged );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void SerializableNeuronLayer::Calculate()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool SerializableNeuronLayer::InitializeNetwork()
{
    if ( m_WeightDimensions.size() > 0U )
    {
        return true;
    }
    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void SerializableNeuronLayer::OnWeightDimensionsVectorChanged( codeframe::PropertyBase* prop )
{
    if ( InitializeNetwork() == true )
    {
        unsigned int wVecSize( 0U );
        unsigned int ioVecSize( 0U );

        unsigned int curValue( 0U );
        unsigned int prewValue( 0U );

        // Iterate through configured weight vector to determinate size of working vectors
        for ( unsigned int n = 0U; n < m_WeightDimensions.size(); n++ )
        {
            curValue = m_WeightDimensions[ n ];
            if ( curValue > ioVecSize)
            {
                ioVecSize = curValue;
            }
            if ( n != 0U )
            {
                wVecSize += prewValue * curValue;
            }
            prewValue = curValue;
        }

        m_WeightVector.resize( wVecSize, 0.0F );
        m_MovingInputLayerVector.resize( ioVecSize, 0.0F );
        m_MovingOutputLayerVector.resize( ioVecSize, 0.0F );

        // Random initialize
        std::mt19937_64 rng;
        // initialize the random number generator with time-dependent seed
        uint64_t timeSeed( std::chrono::high_resolution_clock::now().time_since_epoch().count() );
        std::seed_seq ss
        {
            uint32_t( timeSeed & 0xffffffff ),
            uint32_t( timeSeed >> 32U )
        };
        rng.seed( ss );
        // initialize a uniform distribution between 0 and 1
        std::uniform_real_distribution<double> unif( 0U, 1U );

        for ( unsigned int n = 0U; n < m_WeightVector.size(); n++ )
        {
            m_WeightVector[ n ] = unif( rng );
        }

        LOGGER_DEBUG( LOG_INFO << LOG_LEVEL7 << prop->Path() << " change has triggered Serializable Neuron Layer Initialization wVecSize: " << wVecSize << ", ioVecSize: " << ioVecSize );
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
const std::vector<unsigned int>& SerializableNeuronLayer::GetWeightDimensionsConstVector()
{
    return m_WeightDimensions;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::vector<unsigned int>& SerializableNeuronLayer::GetWeightDimensionsVector()
{
    return m_WeightDimensions;
}
