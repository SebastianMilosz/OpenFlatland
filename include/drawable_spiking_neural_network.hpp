#ifndef DRAWABLE_SPIKING_NEURAL_NETWORK_HPP
#define DRAWABLE_SPIKING_NEURAL_NETWORK_HPP

#include "drawable_object.hpp"
#include "selectable_element.hpp"
#include "spiking_neural_network.hpp"

/*****************************************************************************/
/**
  * @brief Draws the network as a set of nodes placed in model space.
  *
  * Nodes are identified by their index (offset). Every node has a position in
  * model space (NodePosition). Today the positions come from the grid
  * (neighbour distance = 1.0), later they can come from an arbitrary mesh.
  * Drawing, mouse picking and synapse distances all use NodePosition, so only
  * that function has to change when the topology changes.
 **
******************************************************************************/
class DrawableSpikingNeuralNetwork : public SpikingNeuralNetwork, public DrawableObject, public SelectableElement
{
    public:
        using BlockInfo = std::vector<std::tuple<std::string, std::string>>;

                 DrawableSpikingNeuralNetwork(const std::string& name, ObjectNode* parent);
        virtual ~DrawableSpikingNeuralNetwork();

        void Calculate(const thrust::host_vector<float>& dataInput, thrust::host_vector<float>& dataOutput) override;
        void draw( sf::RenderTarget& target, sf::RenderStates states ) const override;

        void    OnHover(int32_t nodeId) override;
        void    OnHoverPixels(float pixelX, float pixelY, float width, float height);
        int32_t GetHovered() const;

        void      Select(int32_t nodeId);
        bool      IsValidNode(int32_t nodeId) const;
        uint32_t  GetNodeCount() const;
        BlockInfo GetBlockInfo(int32_t nodeId) const;

        sf::Vector2f NodePosition(uint32_t nodeId) const;
        float        NodeDistance(uint32_t nodeA, uint32_t nodeB) const;
    private:
        static constexpr float NODE_BOX_SIZE            = 10.0f;
        static constexpr float INPUT_OUTPUT_SLOT_HEIGHT = 10.0f;
        static constexpr float VIEW_MARGIN              = 10.0f; // pixels around the node cloud
        static constexpr float PICK_RADIUS              = 10.0f; // pixels from node center

        // Maps model space positions to screen pixels (uniform scale, so distances are kept)
        struct ViewTransform
        {
            bool         valid  = false;
            float        scale  = 1.0f;
            sf::Vector2f modelMin;
            sf::Vector2f screenOrigin;
        };

        ViewTransform GetView(float width, float height) const;
        sf::Vector2f  ToScreen(const ViewTransform& view, const sf::Vector2f& modelPos) const;
        int32_t       PickNode(const ViewTransform& view, const sf::Vector2f& screenPos) const;

        int32_t m_selected = NO_NODE;
        int32_t m_hovered  = NO_NODE;

        uint32_t GetSynapseSize() const { return m_Synapse.Size; }
        codeframe::Point2D<unsigned int> OffsetToCoordinate(const uint32_t offset) const;

        thrust::host_vector<float> m_dataInput;
        thrust::host_vector<float> m_dataOutput;
};

#endif // DRAWABLE_SPIKING_NEURAL_NETWORK_HPP
