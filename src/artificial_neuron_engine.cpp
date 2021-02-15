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
    CellPoolSize(this, "CellPoolSize", Point2D<unsigned int>( 10U, 10U ), cPropertyInfo().Kind( KIND_2DPOINT, KIND_NUMBER ).Description("CellPoolSize")),
    Input       (this, "Input"       , thrust::host_vector<float>()     , cPropertyInfo().Kind( KIND_VECTOR_THRUST_HOST, KIND_REAL ).Description("Input") , [this]() -> const thrust::host_vector<float>& { return this->m_vectInData; }),
    Output      (this, "Output"      , thrust::host_vector<float>()     , cPropertyInfo().Kind( KIND_VECTOR_THRUST_HOST, KIND_REAL ).Description("Output"), [this]() -> const thrust::host_vector<float>& { return this->m_vectOutData; }),
    m_Inputs("NeuronInputs", this),
    m_Outputs("NeuronOutputs", this),
    m_NeuronCellPool("NeuronCellPool", this, m_vectInData, m_vectOutData),
    m_populateDelay(0U)
{
    CellPoolSize.signalChanged.connect( this, &ArtificialNeuronEngine::OnCellPoolSize );

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

    m_NeuronCellPool.Calculate(m_inputData, m_outputData);

    if (m_populateDelay > 70)
    {
        m_populateDelay = 0U;
        m_NeuronCellPool.Populate();
    }
    else
    {
        m_populateDelay++;
    }

    ProcesseOutputs();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ArtificialNeuronEngine::OnCellPoolSize(codeframe::PropertyNode* prop)
{
    auto propSize = dynamic_cast< codeframe::Property< codeframe::Point2D<unsigned int> >* >(prop);
    if (propSize)
    {
        m_NeuronCellPool.Initialize();
    }
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
