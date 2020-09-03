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
    CellPoolSize              (this, "CellPoolSize"              , 100U                           , cPropertyInfo().Kind( KIND_NUMBER ).Description("CellPoolSize")),
    Input                     (this, "Input"                     , thrust::host_vector<float>()   , cPropertyInfo().Kind( KIND_VECTOR_THRUST_HOST, KIND_REAL ).Description("Input") , [this]() -> const thrust::host_vector<float>& { return this->m_vectInData; }),
    Output                    (this, "Output"                    , thrust::host_vector<float>()   , cPropertyInfo().Kind( KIND_VECTOR_THRUST_HOST, KIND_REAL ).Description("Output"), [this]() -> const thrust::host_vector<float>& { return this->m_vectOutData; }),
    m_Inputs("NeuronInputs", this),
    m_Outputs("NeuronOutputs", this),
    m_NeuronCellPool("NeuronCellPool", this)
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

    m_NeuronCellPool.Calculate();
    m_NeuronCellPool.Populate();

    ProcesseOutputs();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ArtificialNeuronEngine::CollectInputs()
{
    m_vectInData.clear();

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
    m_vectOutData.clear();

    for ( unsigned int n = 0U; n < m_Outputs.Count(); n++ )
    {
        smart_ptr<NeuronLayer> neuronLayerObj = smart_dynamic_pointer_cast<NeuronLayer>(m_Outputs.Get( n ));

        if (smart_ptr_isValid(neuronLayerObj))
        {
            neuronLayerObj->ProcessData(m_vectOutData);
        }
    }
}
