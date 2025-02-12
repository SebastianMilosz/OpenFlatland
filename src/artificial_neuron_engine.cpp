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
    Input(this, "Input" , thrust::host_vector<float>(), cPropertyInfo().Kind( KIND_VECTOR_THRUST_HOST, KIND_REAL ).
                                                                        Description("Input") , [this]() -> const thrust::host_vector<float>&
                                                                        {
                                                                            return this->m_vectInData;
                                                                        }),
    Output(this, "Output", thrust::host_vector<float>(), cPropertyInfo().Kind( KIND_VECTOR_THRUST_HOST, KIND_REAL ).
                                                                         Description("Output"), [this]() -> const thrust::host_vector<float>&
                                                                        {
                                                                            return this->m_vectOutData;
                                                                        }),
    m_InputsObjects("NeuronInputsObjects", this),
    m_OutputsObjects("NeuronOutputsObjects", this),
    m_NeuronCellPool("NeuronCellPool", this)
{
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ArtificialNeuronEngine::Calculate()
{
    CollectInputs();
    m_NeuronCellPool.Calculate(m_vectInData, m_vectOutData);
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

    for ( unsigned int n = 0U; n < m_InputsObjects.Count(); n++ )
    {
        smart_ptr<NeuronLayer> neuronLayerObj = smart_dynamic_pointer_cast<NeuronLayer>(m_InputsObjects.Get( n ));

        if (smart_ptr_isValid(neuronLayerObj))
        {
            neuronLayerObj->GiveData(m_vectInData);
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
    uint32_t vectPos = 0U;
    uint32_t vectSize = 0U;
    for ( uint32_t n = 0U; n < m_OutputsObjects.Count(); n++ )
    {
        smart_ptr<NeuronLayer> neuronLayerObj = smart_dynamic_pointer_cast<NeuronLayer>(m_OutputsObjects.Get( n ));

        if (smart_ptr_isValid(neuronLayerObj))
        {
            vectPos += neuronLayerObj->TakeData(m_vectOutData, vectPos);
            vectSize += neuronLayerObj->size();
        }
    }

    if (m_vectOutData.size() != vectSize)
    {
        m_vectOutData.resize(vectSize);
    }
}
