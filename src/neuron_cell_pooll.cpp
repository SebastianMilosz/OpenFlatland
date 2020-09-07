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
    //ctor
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
void NeuronCellPool::Initialize(const uint32_t cnt)
{
    uint32_t currentSize = m_IntegrateLevel.size();

    if (currentSize != cnt)
    {
        uint32_t slim  = (uint32_t)NeuronSynapseLimit;
        uint32_t solim = (uint32_t)NeuronOutputLimit;

        if (slim > MAX_SYNAPSE_CNT)
        {
            slim = MAX_SYNAPSE_CNT;
            NeuronSynapseLimit = slim;
        }
        if (solim > MAX_OUTPUT_CNT)
        {
            solim = MAX_OUTPUT_CNT;
            NeuronOutputLimit = solim;
        }

        thrust::host_vector<uint32_t> newSynapseLink(slim * cnt, 0U);
        thrust::host_vector<float>    newSynapseWeight(slim * cnt, 0.0f);
        thrust::host_vector<uint32_t> newIntegrateLevel(cnt, 0.0f);
        thrust::host_vector<uint32_t> newIntegrateThreshold(cnt, 0.0f);
        thrust::host_vector<bool>     newOutput(solim * cnt, false);

        m_Synapse.Link = newSynapseLink;
        m_Synapse.Weight = newSynapseWeight;
        m_IntegrateLevel = newIntegrateLevel;
        m_IntegrateThreshold = newIntegrateThreshold;
        m_Output = newOutput;

        m_CurrentSize = currentSize;
        m_CurrentSynapseLimit = slim;
        m_CurrentOutputLimit = solim;
    }
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
