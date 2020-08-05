#include "artificial_neuron_engine.hpp"
#include "neuron_layer_ray.hpp"
#include "neuron_layer_vector.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ArtificialNeuronEngine::ArtificialNeuronEngine( const std::string& name, ObjectNode* parent ) :
    Object( name, parent ),
    m_Inputs("NeuronInputs", this),
    m_Outputs("NeuronOutputs", this)
{
    m_Inputs.Create ( "NeuronLayerVector", "InterfaceEnergyLayer", std::vector<VariantValue>(1U, VariantValue("href", "../../../Energy.EnergyVector")) );
    m_Inputs.Create ( "NeuronLayerRay"   , "InterfaceVisionLayer", std::vector<VariantValue>(1U, VariantValue("href", "../../../Vision.VisionVector")) );
    m_Outputs.Create( "NeuronLayerVector", "InterfaceMotionLayer", std::vector<VariantValue>(1U, VariantValue("href", "../../../Motion.MotionVector>")) );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ArtificialNeuronEngine::Calculate()
{
    CollectInputs();

    ProcesseOutputs();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ArtificialNeuronEngine::CollectInputs()
{
    for ( unsigned int n = 0U; n < m_Inputs.Count(); n++ )
    {
        smart_ptr<NeuronLayer> neuronLayerObj = smart_dynamic_pointer_cast<NeuronLayer>(m_Inputs.Get( n ));

        if (smart_ptr_isValid(neuronLayerObj))
        {
            neuronLayerObj->ProcessData(m_vectInData);
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ArtificialNeuronEngine::ProcesseOutputs()
{
    for ( unsigned int n = 0U; n < m_Outputs.Count(); n++ )
    {
        smart_ptr<NeuronLayer> neuronLayerObj = smart_dynamic_pointer_cast<NeuronLayer>(m_Outputs.Get( n ));

        if (smart_ptr_isValid(neuronLayerObj))
        {
            neuronLayerObj->ProcessData(m_vectOutData);
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ArtificialNeuronEngine::OnWeightDimensionsVectorChanged( codeframe::PropertyNode* prop )
{
    /*
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
    */
}
