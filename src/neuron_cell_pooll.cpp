#include "neuron_cell_pool.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
NeuronCellPool::NeuronCellPool( const std::string& name, ObjectNode* parent ) :
    Object( name, parent ),
    CellPoolSynapseLink       (this, "CellPoolSynapseLink"       , thrust::host_vector<uint32_t>(), cPropertyInfo().Kind( KIND_VECTOR_THRUST_HOST, KIND_NUMBER ).Description("CellPoolSynapseLink")),
    CellPoolSynapseWeight     (this, "CellPoolSynapseWeight"     , thrust::host_vector<float>()   , cPropertyInfo().Kind( KIND_VECTOR_THRUST_HOST, KIND_REAL ).Description("CellPoolSynapseWeight")),
    CellPoolOffsetSynapse     (this, "CellPoolOffsetSynapse"     , thrust::host_vector<uint32_t>(), cPropertyInfo().Kind( KIND_VECTOR_THRUST_HOST, KIND_NUMBER ).Description("CellPoolOffsetSynapse")),
    CellPoolOffsetWeight      (this, "CellPoolOffsetWeight"      , thrust::host_vector<uint32_t>(), cPropertyInfo().Kind( KIND_VECTOR_THRUST_HOST, KIND_NUMBER ).Description("CellPoolOffsetWeight")),
    CellPoolOffsetOutput      (this, "CellPoolOffsetOutput"      , thrust::host_vector<uint32_t>(), cPropertyInfo().Kind( KIND_VECTOR_THRUST_HOST, KIND_NUMBER ).Description("CellPoolOffsetOutput")),
    CellPoolSizeSynapse       (this, "CellPoolSizeSynapse"       , thrust::host_vector<uint32_t>(), cPropertyInfo().Kind( KIND_VECTOR_THRUST_HOST, KIND_NUMBER ).Description("CellPoolSizeSynapse"),
                                [this]() -> const thrust::host_vector<uint32_t>& { return this->m_Size.Synapse; }
                              ),
    CellPoolSizeWeight        (this, "CellPoolSizeWeight"        , thrust::host_vector<uint32_t>(), cPropertyInfo().Kind( KIND_VECTOR_THRUST_HOST, KIND_NUMBER ).Description("CellPoolSizeWeight"),
                                [this]() -> const thrust::host_vector<uint32_t>& { return this->m_Size.Weight; }
                              ),
    CellPoolSizeOutput        (this, "CellPoolSizeOutput"        , thrust::host_vector<uint32_t>(), cPropertyInfo().Kind( KIND_VECTOR_THRUST_HOST, KIND_NUMBER ).Description("CellPoolSizeOutput"),
                                [this]() -> const thrust::host_vector<uint32_t>& { return this->m_Size.Output; }
                              ),
    CellPoolIntegrateThreshold(this, "CellPoolIntegrateThreshold", thrust::host_vector<float>()   , cPropertyInfo().Kind( KIND_VECTOR_THRUST_HOST, KIND_REAL ).Description("CellPoolIntegrateThreshold"),
                                [this]() -> const thrust::host_vector<float>& { return this->m_IntegrateThreshold; }
                              ),
    CellPoolIntegrateLevel    (this, "CellPoolIntegrateLevel"    , thrust::host_vector<float>()   , cPropertyInfo().Kind( KIND_VECTOR_THRUST_HOST, KIND_REAL ).Description("CellPoolIntegrateLevel"),
                                [this]() -> const thrust::host_vector<float>& { return this->m_IntegrateLevel; }
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
