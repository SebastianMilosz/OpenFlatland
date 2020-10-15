#ifndef ARTIFICIAL_NEURON_ENGINE_HPP_INCLUDED
#define ARTIFICIAL_NEURON_ENGINE_HPP_INCLUDED

#include <sigslot.h>
#include <serializable_object_node.hpp>
#include <serializable_object.hpp>

#include "neuron_layer_container.hpp"
#include "neuron_cell_pool.hpp"

class ArtificialNeuronEngine : public codeframe::Object
{
        CODEFRAME_META_CLASS_NAME( "ArtificialNeuronEngine" );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
                 ArtificialNeuronEngine( const std::string& name, ObjectNode* parent );
        virtual ~ArtificialNeuronEngine() = default;

        codeframe::Property< codeframe::Point2D<unsigned int> > CellPoolSize;
        codeframe::Property< thrust::host_vector<float> > Input;
        codeframe::Property< thrust::host_vector<float> > Output;

        void Calculate();

        void OnCellPoolSize(codeframe::PropertyNode* prop);

        NeuronCellPool& GetPool() {return m_NeuronCellPool;}

    protected:
        NeuronLayerContainer m_Inputs;
        NeuronLayerContainer m_Outputs;

        NeuronCellPool m_NeuronCellPool;

        virtual void CollectInputs();
        virtual void ProcesseOutputs();

    private:
        thrust::host_vector<float> m_vectInData;
        thrust::host_vector<float> m_vectOutData;

        uint32_t m_populateDelay;
};

#endif // ARTIFICIAL_NEURON_ENGINE_HPP_INCLUDED
