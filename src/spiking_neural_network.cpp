#include "spiking_neural_network.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
SpikingNeuralNetwork::SpikingNeuralNetwork(const std::string& name, ObjectNode* parent) :
    Model_SNN(name, parent),
    CellPoolSize      (this, "CellPoolSize", Point2D<unsigned int>( 10U, 10U ), cPropertyInfo().Kind( KIND_2DPOINT, KIND_NUMBER ).Description("CellPoolSize")),
    NeuronSynapseLimit(this, "NeuronSynapseLimit", 100U, cPropertyInfo().Kind( KIND_NUMBER ).Description("NeuronSynapseLimit")),
    NeuronOutputLimit (this, "NeuronOutputLimit", 10U, cPropertyInfo().Kind( KIND_NUMBER ).Description("NeuronOutputLimit")),
    SynapseLink       (this, "SynapseLink"       , thrust::host_vector<float>(), cPropertyInfo().GUIMode( GUIMODE_DISABLED ).Kind( KIND_VECTOR_THRUST_HOST, KIND_REAL ).Description("SynapseLink"),
                        [this]() -> const thrust::host_vector<float>& { return this->m_Synapse.Link; },
                        nullptr,
                        [this]() -> thrust::host_vector<float>& { return this->m_Synapse.Link; }
                      ),
    SynapseWeight     (this, "SynapseWeight"     , thrust::host_vector<float>()   , cPropertyInfo().GUIMode( GUIMODE_DISABLED ).Kind( KIND_VECTOR_THRUST_HOST, KIND_REAL ).Description("SynapseWeight"),
                        [this]() -> const thrust::host_vector<float>& { return this->m_Synapse.Weight; },
                        nullptr,
                        [this]() -> thrust::host_vector<float>& { return this->m_Synapse.Weight; }
                      ),
    IntegrateThreshold(this, "IntegrateThreshold", thrust::host_vector<float>()   , cPropertyInfo().GUIMode( GUIMODE_DISABLED ).Kind( KIND_VECTOR_THRUST_HOST, KIND_REAL ).Description("IntegrateThreshold"),
                        [this]() -> const thrust::host_vector<float>& { return this->m_IntegrateThreshold; },
                        nullptr,
                        [this]() -> thrust::host_vector<float>& { return this->m_IntegrateThreshold; }
                      ),
    IntegrateLevel    (this, "IntegrateLevel"    , thrust::host_vector<float>()   , cPropertyInfo().GUIMode( GUIMODE_DISABLED ).Kind( KIND_VECTOR_THRUST_HOST, KIND_REAL ).Description("IntegrateLevel"),
                        [this]() -> const thrust::host_vector<float>& { return this->m_IntegrateLevel; },
                        nullptr,
                        [this]() -> thrust::host_vector<float>& { return this->m_IntegrateLevel; }
                      ),
    m_generator(std::random_device()()),
    m_populateDelay(0U)
{
    NeuronSynapseLimit.signalChanged.connect( this, &SpikingNeuralNetwork::OnNeuronSynapseLimit );
    NeuronOutputLimit.signalChanged.connect( this, &SpikingNeuralNetwork::OnNeuronOutputLimit );
    CellPoolSize.signalChanged.connect( this, &SpikingNeuralNetwork::OnCellPoolSize );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
SpikingNeuralNetwork::~SpikingNeuralNetwork()
{
    //dtor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void SpikingNeuralNetwork::OnNeuronSynapseLimit(codeframe::PropertyNode* prop)
{
    //Initialize(m_CurrentSize);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void SpikingNeuralNetwork::OnNeuronOutputLimit(codeframe::PropertyNode* prop)
{
    //Initialize(m_CurrentSize);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void SpikingNeuralNetwork::Resize(const uint32_t width, const uint32_t height)
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void SpikingNeuralNetwork::Initialize(unsigned int w, unsigned int h)
{
    m_CurrentSize = codeframe::Point2D<unsigned int>(w,h);

    const uint32_t newSize = m_CurrentSize.X() * m_CurrentSize.Y();
    const uint32_t currentSize = m_IntegrateLevel.size();
    const uint32_t slim  = std::min((uint32_t)NeuronSynapseLimit, static_cast<uint32_t>(MAX_SYNAPSE_CNT));

    if (currentSize != newSize || slim != m_Synapse.Size)
    {
        thrust::host_vector<float>    newSynapseLink  (slim * newSize, 0.0f);
        thrust::host_vector<float>    newSynapseWeight(slim * newSize, 0.0f);
        thrust::host_vector<float>    newIntegrateLevel(newSize, 0.0f);
        thrust::host_vector<float>    newIntegrateThreshold(newSize, 0.0f);
        thrust::host_vector<uint32_t> newIntegrateInterval(newSize, 0U);
        thrust::host_vector<uint64_t> newOutput(newSize, 0U);

        thrust::for_each(m_Synapse.Link.begin()      , m_Synapse.Link.end()      , copy_range_functor<float>(newSynapseLink       , slim));
        thrust::for_each(m_Synapse.Weight.begin()    , m_Synapse.Weight.end()    , copy_range_functor<float>(newSynapseWeight     , slim));
        thrust::for_each(m_Output.begin()            , m_Output.end()            , copy_range_functor<uint64_t>(newOutput         , slim));
        thrust::for_each(m_IntegrateLevel.begin()    , m_IntegrateLevel.end()    , copy_range_functor<float>(newIntegrateLevel    , 1U));
        thrust::for_each(m_IntegrateThreshold.begin(), m_IntegrateThreshold.end(), copy_range_functor<float>(newIntegrateThreshold, 1U));
        thrust::for_each(m_IntegrateInterval.begin() , m_IntegrateInterval.end() , copy_range_functor<uint32_t>(newIntegrateInterval, 1U));

        thrust::fill(newIntegrateThreshold.begin(), newIntegrateThreshold.end(), 10.0f);

        m_Synapse.Link = newSynapseLink;
        m_Synapse.Weight = newSynapseWeight;
        m_IntegrateLevel = newIntegrateLevel;
        m_IntegrateThreshold = newIntegrateThreshold;
        m_IntegrateInterval = newIntegrateInterval;
        m_Output = newOutput;
    }

    NeuronSynapseLimit = m_Synapse.Size = slim;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void SpikingNeuralNetwork::OnCellPoolSize(codeframe::PropertyNode* prop)
{
    auto propSize = dynamic_cast< codeframe::Property< codeframe::Point2D<unsigned int> >* >(prop);
    if (propSize)
    {
        Initialize(propSize->GetConstValue().X(), propSize->GetConstValue().Y());
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void SpikingNeuralNetwork::Calculate(const thrust::host_vector<float>& dataInput,
                                     thrust::host_vector<float>& dataOutput)
{
    const uint32_t poolSize = m_CurrentSize.X() * m_CurrentSize.Y();

    thrust::counting_iterator<uint32_t> first(0U);
    thrust::counting_iterator<uint32_t> last = first + poolSize;

    // Synthesize all synapse inputs into one value for each neuron
    thrust::for_each(
                     thrust::make_zip_iterator(thrust::make_tuple(
                                                                  first,
                                                                  m_IntegrateLevel.begin()
                                                                 )),
                     thrust::make_zip_iterator(thrust::make_tuple(
                                                                  last,
                                                                  m_IntegrateLevel.end()
                                                                 )),
                     neuron_calculate_functor(m_Output, m_Synapse, dataInput)
                    );

    // Calculate and propagate it through output
    thrust::for_each(
                     thrust::make_zip_iterator(thrust::make_tuple(
                                                                  m_IntegrateLevel.begin(),
                                                                  m_IntegrateThreshold.begin(),
                                                                  m_IntegrateInterval.begin(),
                                                                  m_Output.begin()
                                                                 )),
                     thrust::make_zip_iterator(thrust::make_tuple(
                                                                  m_IntegrateLevel.end(),
                                                                  m_IntegrateThreshold.end(),
                                                                  m_IntegrateInterval.end(),
                                                                  m_Output.end()
                                                                 )),
                     neuron_output_calculate_functor()
                    );

    // Outputs vector connection
    thrust::transform(
                        dataOutput.begin(),
                        dataOutput.end(),
                        dataOutput.begin(),
                        neuron_output_take_functor<float>(m_IntegrateLevel)
                     );

    if (m_populateDelay > 2)
    {
        m_populateDelay = 0U;
        Populate(dataInput, dataOutput);
    }
    else
    {
        m_populateDelay++;
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void SpikingNeuralNetwork::Populate(const thrust::host_vector<float>& dataInput,
                                    thrust::host_vector<float>& dataOutput)
{
    const uint32_t poolSize = m_CurrentSize.X() * m_CurrentSize.Y();

    thrust::counting_iterator<uint32_t> first(0U);
    thrust::counting_iterator<uint32_t> last = first + poolSize;

    // Synthesize all synapse inputs into one value for each neuron
    thrust::for_each(
                     thrust::make_zip_iterator(thrust::make_tuple(
                                                                  first,
                                                                  m_IntegrateLevel.begin()
                                                                 )),
                     thrust::make_zip_iterator(thrust::make_tuple(
                                                                  last,
                                                                  m_IntegrateLevel.end()
                                                                 )),
                     neuron_populate_functor(m_Output, m_Synapse, dataInput, m_CurrentSize, m_generator)
                    );
}
