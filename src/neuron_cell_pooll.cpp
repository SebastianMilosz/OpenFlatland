#include "neuron_cell_pool.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
NeuronCellPool::NeuronCellPool( const std::string& name, ObjectNode* parent ) :
    Object( name, parent ),
    NeuronSynapseLimit(this, "NeuronSynapseLimit", 100U, cPropertyInfo().Kind( KIND_NUMBER ).Description("NeuronSynapseLimit")),
    NeuronOutputLimit (this, "NeuronOutputLimit", 10U, cPropertyInfo().Kind( KIND_NUMBER ).Description("NeuronOutputLimit")),
    SynapseLink       (this, "SynapseLink"       , thrust::host_vector<uint32_t>(), cPropertyInfo().GUIMode( GUIMODE_DISABLED ).Kind( KIND_VECTOR_THRUST_HOST, KIND_NUMBER ).Description("SynapseLink"),
                        [this]() -> const thrust::host_vector<uint32_t>& { return this->m_Synapse.Link; },
                        nullptr,
                        [this]() -> thrust::host_vector<uint32_t>& { return this->m_Synapse.Link; }
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
    m_distribution(1)
{
    NeuronSynapseLimit.signalChanged.connect( this, &NeuronCellPool::OnNeuronSynapseLimit );
    NeuronOutputLimit.signalChanged.connect( this, &NeuronCellPool::OnNeuronOutputLimit );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
NeuronCellPool::~NeuronCellPool()
{
    //dtor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void NeuronCellPool::OnNeuronSynapseLimit(codeframe::PropertyNode* prop)
{
    Initialize(m_IntegrateLevel.size());
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void NeuronCellPool::OnNeuronOutputLimit(codeframe::PropertyNode* prop)
{
    Initialize(m_IntegrateLevel.size());
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void NeuronCellPool::Initialize(const uint32_t cnt)
{
    const uint32_t currentSize = m_IntegrateLevel.size();
    const uint32_t slim  = std::min((uint32_t)NeuronSynapseLimit, static_cast<uint32_t>(MAX_SYNAPSE_CNT));
    const uint32_t solim = std::min((uint32_t)NeuronOutputLimit, static_cast<uint32_t>(MAX_OUTPUT_CNT));

    if (currentSize != cnt || slim != m_CurrentSynapseLimit || solim != m_CurrentOutputLimit)
    {
        thrust::host_vector<uint32_t> newSynapseLink(slim * cnt, 0U);
        thrust::host_vector<float>    newSynapseWeight(slim * cnt, 0.0f);
        thrust::host_vector<float>    newIntegrateLevel(cnt, 0.0f);
        thrust::host_vector<float>    newIntegrateThreshold(cnt, 0.0f);
        thrust::host_vector<bool>     newOutput(solim * cnt, false);

        thrust::for_each(m_Synapse.Link.begin()      , m_Synapse.Link.end()      , copy_range_functor<uint32_t>(newSynapseLink       , slim));
        thrust::for_each(m_Synapse.Weight.begin()    , m_Synapse.Weight.end()    , copy_range_functor<float>   (newSynapseWeight     , slim));
        thrust::for_each(m_Output.begin()            , m_Output.end()            , copy_range_functor<bool>    (newOutput            , solim));
        thrust::for_each(m_IntegrateLevel.begin()    , m_IntegrateLevel.end()    , copy_range_functor<float>   (newIntegrateLevel    , 1U));
        thrust::for_each(m_IntegrateThreshold.begin(), m_IntegrateThreshold.end(), copy_range_functor<float>   (newIntegrateThreshold, 1U));

        m_Synapse.Link = newSynapseLink;
        m_Synapse.Weight = newSynapseWeight;
        m_IntegrateLevel = newIntegrateLevel;
        m_IntegrateThreshold = newIntegrateThreshold;
        m_Output = newOutput;
    }

    NeuronSynapseLimit = m_CurrentSynapseLimit = slim;
    NeuronOutputLimit  = m_CurrentOutputLimit = solim;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void NeuronCellPool::Calculate()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void NeuronCellPool::Populate()
{

}
