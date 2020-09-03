#ifndef NEURON_CELL_POOL_HPP
#define NEURON_CELL_POOL_HPP

#include <random>
#include <sigslot.h>
#include <serializable_object_node.hpp>
#include <serializable_object.hpp>

#include <thrust/device_vector.h>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
class NeuronCellPool : public codeframe::Object
{
    CODEFRAME_META_CLASS_NAME( "NeuronCellPool" );
    CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
        // Those vectors are used to store current neuron pool state in nvs
        codeframe::Property< thrust::host_vector<uint32_t> > CellPoolSynapseLink;
        codeframe::Property< thrust::host_vector<float> >    CellPoolSynapseWeight;
        codeframe::Property< thrust::host_vector<uint32_t> > CellPoolOffsetSynapse;
        codeframe::Property< thrust::host_vector<uint32_t> > CellPoolOffsetWeight;
        codeframe::Property< thrust::host_vector<uint32_t> > CellPoolOffsetOutput;
        codeframe::Property< thrust::host_vector<uint32_t> > CellPoolSizeSynapse;
        codeframe::Property< thrust::host_vector<uint32_t> > CellPoolSizeWeight;
        codeframe::Property< thrust::host_vector<uint32_t> > CellPoolSizeOutput;
        codeframe::Property< thrust::host_vector<float> >    CellPoolIntegrateThreshold;
        codeframe::Property< thrust::host_vector<float> >    CellPoolIntegrateLevel;

        struct SynapseVector
        {
            thrust::host_vector<uint32_t> Link;
            thrust::host_vector<float>    Weight;
        };

        struct OffsetVector
        {
            thrust::host_vector<uint32_t> Synapse;
            thrust::host_vector<uint32_t> Weight;
            thrust::host_vector<uint32_t> Output;
        };

        struct SizeVector
        {
            thrust::host_vector<uint32_t> Synapse;
            thrust::host_vector<uint32_t> Weight;
            thrust::host_vector<uint32_t> Output;
        };

        NeuronCellPool( const std::string& name, ObjectNode* parent );
        virtual ~NeuronCellPool();

        void Initialize()
        {

        }

        void Calculate()
        {

        }

        void Populate()
        {

        }

    private:
        OffsetVector               m_Offset;
        SizeVector                 m_Size;

        SynapseVector              m_Synapse;
        thrust::host_vector<float> m_IntegrateLevel;
        thrust::host_vector<float> m_IntegrateThreshold;
        thrust::host_vector<bool>  m_Output;

        std::mt19937 m_generator;
        std::exponential_distribution<> m_distribution;
};

#endif // NEURON_CELL_POOL_HPP
