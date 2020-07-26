#ifndef NEURON_LAYER_VECTOR_HPP_INCLUDED
#define NEURON_LAYER_VECTOR_HPP_INCLUDED

#include <sigslot.h>

#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/sort.h>

#include "neuron_layer.hpp"

class NeuronLayerVector : public NeuronLayer
{
    CODEFRAME_META_CLASS_NAME( "NeuronLayerVector" );
    CODEFRAME_META_BUILD_TYPE( codeframe::DYNAMIC );

    public:
                 NeuronLayerVector( const std::string& name, ObjectNode* parent, const std::string& link );
        virtual ~NeuronLayerVector() = default;

        codeframe::Property< thrust::host_vector<float> > Input;

        void Calculate() override;

};

#endif // NEURON_LAYER_VECTOR_HPP_INCLUDED
