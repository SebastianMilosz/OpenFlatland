#ifndef DRAWABLE_SPIKING_NEURAL_NETWORK_HPP
#define DRAWABLE_SPIKING_NEURAL_NETWORK_HPP

#include "drawable_object.hpp"
#include "spiking_neural_network.hpp"

#include <imgui.h>
#include <imgui-SFML.h>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
class DrawableSpikingNeuralNetwork : public SpikingNeuralNetwork, public DrawableObject
{
    public:
                 DrawableSpikingNeuralNetwork(const std::string& name, ObjectNode* parent);
        virtual ~DrawableSpikingNeuralNetwork();

        void Calculate(const thrust::host_vector<float>& dataInput, thrust::host_vector<float>& dataOutput) override;
        void draw( sf::RenderTarget& target, sf::RenderStates states ) const override;

        void Select(uint32_t x, uint32_t y);
    private:
        const uint32_t neuronBoxW = 10U;
        const uint32_t neuronBoxH = 10U;

        uint32_t GetSynapseSize() const { return m_Synapse.Size; }
        uint32_t CoordinateToOffset(const uint32_t x, const uint32_t y) const;
        codeframe::Point2D<unsigned int> OffsetToCoordinate(const uint32_t offset) const;

        mutable sf::Vertex line[2] =
        {
            sf::Vertex(sf::Vector2f()),
            sf::Vertex(sf::Vector2f())
        };

        thrust::host_vector<float> m_dataInput;
        mutable sf::Text           m_text;

        int32_t m_selectX = 0;
        int32_t m_selectY = 0;
};

#endif // DRAWABLE_SPIKING_NEURAL_NETWORK_HPP
