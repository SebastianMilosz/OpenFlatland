#include "drawable_spiking_neural_network.hpp"

#include "colorize_number.hpp"

#include <MathUtilities.h>

#include <algorithm>
#include <cmath>
#include <limits>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
DrawableSpikingNeuralNetwork::DrawableSpikingNeuralNetwork(const std::string& name, ObjectNode* parent) :
    SpikingNeuralNetwork(name, parent)
{
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
DrawableSpikingNeuralNetwork::~DrawableSpikingNeuralNetwork()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void DrawableSpikingNeuralNetwork::Calculate(const thrust::host_vector<float>& dataInput, thrust::host_vector<float>& dataOutput)
{
    m_dataInput = dataInput;
    SpikingNeuralNetwork::Calculate(dataInput, dataOutput);
    m_dataOutput = dataOutput;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
uint32_t DrawableSpikingNeuralNetwork::GetNodeCount() const
{
    return m_CurrentSize.X() * m_CurrentSize.Y();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool DrawableSpikingNeuralNetwork::IsValidNode(int32_t nodeId) const
{
    return (nodeId >= 0) && (static_cast<uint32_t>(nodeId) < GetNodeCount());
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void DrawableSpikingNeuralNetwork::Select(int32_t nodeId)
{
    m_selected = IsValidNode(nodeId) ? nodeId : NO_NODE;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void DrawableSpikingNeuralNetwork::OnHover(int32_t nodeId)
{
    m_hovered = IsValidNode(nodeId) ? nodeId : NO_NODE;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
int32_t DrawableSpikingNeuralNetwork::GetHovered() const
{
    return IsValidNode(m_hovered) ? m_hovered : NO_NODE;
}

/*****************************************************************************/
/**
  * @brief Position of the node in model space.
  *        Currently derived from the grid layout, neighbour distance is 1.0.
  *        Replace this when nodes get their own (mesh) positions.
 **
******************************************************************************/
sf::Vector2f DrawableSpikingNeuralNetwork::NodePosition(uint32_t nodeId) const
{
    const codeframe::Point2D<unsigned int> coord = OffsetToCoordinate(nodeId);
    return {static_cast<float>(coord.X()), static_cast<float>(coord.Y())};
}

/*****************************************************************************/
/**
  * @brief Euclidean distance between two nodes in model space
 **
******************************************************************************/
float DrawableSpikingNeuralNetwork::NodeDistance(uint32_t nodeA, uint32_t nodeB) const
{
    const sf::Vector2f d = NodePosition(nodeA) - NodePosition(nodeB);
    return std::sqrt(d.x * d.x + d.y * d.y);
}

/*****************************************************************************/
/**
  * @brief Fits the bounding box of all nodes into the drawing area between
  *        input and output slots, keeping the aspect ratio.
 **
******************************************************************************/
DrawableSpikingNeuralNetwork::ViewTransform DrawableSpikingNeuralNetwork::GetView(float width, float height) const
{
    ViewTransform view;
    const uint32_t nodeCount = GetNodeCount();

    if (nodeCount == 0U)
    {
        return view;
    }

    sf::Vector2f minPos = NodePosition(0U);
    sf::Vector2f maxPos = minPos;

    for (uint32_t n = 1U; n < nodeCount; n++)
    {
        const sf::Vector2f p = NodePosition(n);
        minPos.x = std::min(minPos.x, p.x);
        minPos.y = std::min(minPos.y, p.y);
        maxPos.x = std::max(maxPos.x, p.x);
        maxPos.y = std::max(maxPos.y, p.y);
    }

    const float areaTop    = INPUT_OUTPUT_SLOT_HEIGHT + VIEW_MARGIN;
    const float areaWidth  = width  - 2.0f * VIEW_MARGIN;
    const float areaHeight = height - 2.0f * (INPUT_OUTPUT_SLOT_HEIGHT + VIEW_MARGIN);

    if (areaWidth <= 0.0f || areaHeight <= 0.0f)
    {
        return view;
    }

    const float extentX = maxPos.x - minPos.x;
    const float extentY = maxPos.y - minPos.y;

    // Uniform scale so on-screen distances stay proportional to model distances
    float scale = std::numeric_limits<float>::max();
    if (extentX > 0.0f) { scale = std::min(scale, areaWidth  / extentX); }
    if (extentY > 0.0f) { scale = std::min(scale, areaHeight / extentY); }
    if (scale == std::numeric_limits<float>::max()) { scale = 1.0f; } // single node

    // Center the node cloud inside the area
    view.scale        = scale;
    view.modelMin     = minPos;
    view.screenOrigin = {VIEW_MARGIN + (areaWidth  - extentX * scale) * 0.5f,
                         areaTop     + (areaHeight - extentY * scale) * 0.5f};
    view.valid        = true;

    return view;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
sf::Vector2f DrawableSpikingNeuralNetwork::ToScreen(const ViewTransform& view, const sf::Vector2f& modelPos) const
{
    return view.screenOrigin + (modelPos - view.modelMin) * view.scale;
}

/*****************************************************************************/
/**
  * @brief Nearest node within PICK_RADIUS pixels, or NO_NODE.
  *        Linear search, good enough for now, use a spatial index for big meshes.
 **
******************************************************************************/
int32_t DrawableSpikingNeuralNetwork::PickNode(const ViewTransform& view, const sf::Vector2f& screenPos) const
{
    int32_t bestNode = NO_NODE;
    float   bestDist2 = PICK_RADIUS * PICK_RADIUS;

    const uint32_t nodeCount = GetNodeCount();
    for (uint32_t n = 0U; n < nodeCount; n++)
    {
        const sf::Vector2f d = ToScreen(view, NodePosition(n)) - screenPos;
        const float dist2 = d.x * d.x + d.y * d.y;

        if (dist2 <= bestDist2)
        {
            bestDist2 = dist2;
            bestNode  = static_cast<int32_t>(n);
        }
    }

    return bestNode;
}

/*****************************************************************************/
/**
  * @brief Converts mouse pixel coordinates (relative to the drawn texture)
  *        to the nearest node and updates hover state
 **
******************************************************************************/
void DrawableSpikingNeuralNetwork::OnHoverPixels(float pixelX, float pixelY, float width, float height)
{
    const ViewTransform view = GetView(width, height);

    if (view.valid == false)
    {
        OnHover(NO_NODE);
        return;
    }

    OnHover(PickNode(view, {pixelX, pixelY}));
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void DrawableSpikingNeuralNetwork::draw( sf::RenderTarget& target, sf::RenderStates states ) const
{
    const thrust::host_vector<float>& synapseLinkVector   = SynapseLink.GetConstValue();
    const thrust::host_vector<float>& synapseWeightVector = SynapseWeight.GetConstValue();
    const thrust::host_vector<float>& integrateLevel      = IntegrateLevel.GetConstValue();

    // If no data is available, skip rendering
    if (m_dataInput.size() == 0U)
    {
        return;
    }

    const float width  = static_cast<float>(target.getSize().x);
    const float height = static_cast<float>(target.getSize().y);
    const float inW    = width / m_dataInput.size();

    sf::RectangleShape rectangle;

    // --- Draw Input Data Layer ---
    float curX = 0.0f;
    for (const auto& value : m_dataInput)
    {
        rectangle.setOutlineThickness(0.0f);
        rectangle.setFillColor(ColorizeNumber_IronBown<float>(value));
        rectangle.setPosition({curX, 0.0f});
        rectangle.setSize({inW, INPUT_OUTPUT_SLOT_HEIGHT});

        target.draw(rectangle, states);
        curX += inW;
    }

    // --- Draw Output Data Layer ---
    if (m_dataOutput.size() > 0U)
    {
        curX = 0.0f;
        const float outW = width / m_dataOutput.size();
        rectangle.setOutlineColor(sf::Color::Blue);

        for (const auto& value : m_dataOutput)
        {
            rectangle.setOutlineThickness(1.0f);
            rectangle.setFillColor(ColorizeNumber_IronBown<float>(value));
            rectangle.setPosition({curX, height - INPUT_OUTPUT_SLOT_HEIGHT});
            rectangle.setSize({outW, INPUT_OUTPUT_SLOT_HEIGHT});

            target.draw(rectangle, states);
            curX += outW;
        }
    }

    const ViewTransform view = GetView(width, height);

    if (view.valid == false)
    {
        return;
    }

    const uint32_t     synapseSize = GetSynapseSize();
    const uint32_t     nodeCount   = GetNodeCount();
    const sf::Vector2f halfBox(NODE_BOX_SIZE * 0.5f, NODE_BOX_SIZE * 0.5f);

    sf::VertexArray synapseLines(sf::PrimitiveType::Lines);
    sf::VertexArray nodeBoxes(sf::PrimitiveType::Triangles);

    for (uint32_t node = 0U; node < nodeCount; node++)
    {
        const float value = (node < integrateLevel.size()) ? integrateLevel[node] : 0.0f;
        const sf::Vector2f center = ToScreen(view, NodePosition(node));

        appendQuad(nodeBoxes, center - halfBox, {NODE_BOX_SIZE, NODE_BOX_SIZE}, ColorizeNumber_IronBown<float>(value));

        // --- Synapses of the current node ---
        for (uint32_t n = 0U; n < synapseSize; n++)
        {
            const uint32_t synapseIndex = node * synapseSize + n;

            if (synapseIndex >= synapseLinkVector.size() || synapseIndex >= synapseWeightVector.size())
            {
                break;
            }

            const float linkValue   = synapseLinkVector[synapseIndex];
            const float weightValue = synapseWeightVector[synapseIndex];

            if (weightValue <= 0.0f)
            {
                continue;
            }

            sf::Vector2f linkPoint;

            if (linkValue > 0.5f)
            {
                // Link to another node
                const uint32_t linkNode = static_cast<uint32_t>(linkValue);
                if (linkNode >= nodeCount)
                {
                    continue;
                }
                linkPoint = ToScreen(view, NodePosition(linkNode));
            }
            else if (linkValue < 0.0f)
            {
                // Link to the input data slot
                linkPoint = {std::floor(std::fabs(linkValue)) * inW + inW * 0.5f, INPUT_OUTPUT_SLOT_HEIGHT};
            }
            else
            {
                break;
            }

            const sf::Color weightColor = ColorizeNumber_IronBown<float>(weightValue);
            synapseLines.append({center, weightColor});
            synapseLines.append({linkPoint, weightColor});
        }
    }

    // Draw synapses and nodes batch
    target.draw(synapseLines, states);
    target.draw(nodeBoxes, states);

    // --- Interaction overlays on top of everything else ---
    sf::RectangleShape overlay;
    overlay.setSize({NODE_BOX_SIZE, NODE_BOX_SIZE});
    overlay.setFillColor(sf::Color::Transparent);

    if (IsValidNode(m_selected))
    {
        overlay.setPosition(ToScreen(view, NodePosition(m_selected)) - halfBox);
        overlay.setOutlineColor(sf::Color::Red);
        overlay.setOutlineThickness(2.0f);
        target.draw(overlay, states);
    }

    if (IsValidNode(m_hovered))
    {
        overlay.setPosition(ToScreen(view, NodePosition(m_hovered)) - halfBox);
        overlay.setOutlineColor(sf::Color::Yellow);
        overlay.setOutlineThickness(1.5f);
        target.draw(overlay, states);
    }
}

/*****************************************************************************/
/**
  * @brief Returns list of (name, value) rows describing the node
 **
******************************************************************************/
DrawableSpikingNeuralNetwork::BlockInfo DrawableSpikingNeuralNetwork::GetBlockInfo(int32_t nodeId) const
{
    BlockInfo retInfo;

    if (IsValidNode(nodeId) == false)
    {
        return retInfo;
    }

    const uint32_t     node     = static_cast<uint32_t>(nodeId);
    const sf::Vector2f position = NodePosition(node);

    retInfo.push_back(std::make_tuple("Node:", "#" + utilities::math::IntToStr(node)));
    retInfo.push_back(std::make_tuple("Position:",
        "(" + utilities::math::FloatToStr(position.x, "%2.2f") + "," +
              utilities::math::FloatToStr(position.y, "%2.2f") + ")"));

    if (node < m_Output.size())
    {
        retInfo.push_back(std::make_tuple("Output:", utilities::math::IntToHex(m_Output[node])));
    }
    if (node < m_IntegrateInterval.size())
    {
        retInfo.push_back(std::make_tuple("Interval:", utilities::math::IntToStr(m_IntegrateInterval[node])));
    }
    if (node < m_IntegrateLevel.size())
    {
        retInfo.push_back(std::make_tuple("Level:", utilities::math::FloatToStr(m_IntegrateLevel[node], "%2.3f")));
    }
    if (node < m_IntegrateThreshold.size())
    {
        retInfo.push_back(std::make_tuple("Threshold:", utilities::math::FloatToStr(m_IntegrateThreshold[node], "%2.3f")));
    }

    retInfo.push_back(std::make_tuple("Synapses:", "value : source w= d="));

    const thrust::host_vector<float>& synapseLinkVector   = SynapseLink.GetConstValue();
    const thrust::host_vector<float>& synapseWeightVector = SynapseWeight.GetConstValue();
    const uint32_t synapseSize = GetSynapseSize();
    const uint32_t nodeCount   = GetNodeCount();

    for (uint32_t n = 0U; n < synapseSize; n++)
    {
        const uint32_t synapseIndex = node * synapseSize + n;

        if (synapseIndex >= synapseLinkVector.size() || synapseIndex >= synapseWeightVector.size())
        {
            break;
        }

        const float linkValue   = synapseLinkVector[synapseIndex];
        const float weightValue = synapseWeightVector[synapseIndex];

        if (weightValue <= 0.0f)
        {
            continue;
        }

        const std::string weightText = " w=" + utilities::math::FloatToStr(weightValue, "%2.3f");
        std::string linkText;
        std::string valueText;

        if (linkValue > 0.0f)
        {
            // Integer part is the source node, fractional part encodes its output bit position
            double intpart;
            const double fracpart = std::modf(linkValue, &intpart);
            const uint32_t linkNode = static_cast<uint32_t>(intpart);

            if (linkNode >= nodeCount || linkNode >= m_Output.size())
            {
                continue;
            }

            const uint8_t bitPos = static_cast<uint8_t>(std::min(63.0, 64.0 * fracpart));
            const bool    bitSet = (m_Output[linkNode] & (static_cast<uint64_t>(1U) << bitPos)) != 0U;

            linkText  = "#" + utilities::math::IntToStr(linkNode) + weightText +
                        " d=" + utilities::math::FloatToStr(NodeDistance(node, linkNode), "%2.2f");
            valueText = utilities::math::FloatToStr(bitSet ? 1.0f : 0.0f, "%2.3f") + " :";
        }
        else if (linkValue < 0.0f)
        {
            const uint32_t inPos = static_cast<uint32_t>(std::fabs(linkValue));
            const float    value = (inPos < m_dataInput.size()) ? m_dataInput[inPos] : 0.0f;

            linkText  = "in[" + utilities::math::IntToStr(inPos) + "]" + weightText;
            valueText = utilities::math::FloatToStr(value, "%2.3f") + " :";
        }
        else
        {
            break;
        }

        retInfo.push_back(std::make_tuple(valueText, linkText));
    }

    return retInfo;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
codeframe::Point2D<unsigned int> DrawableSpikingNeuralNetwork::OffsetToCoordinate(const uint32_t offset) const
{
    codeframe::Point2D<unsigned int> retValue;

    if (m_CurrentSize.X() == 0U)
    {
        return retValue;
    }

    retValue.SetX(offset % m_CurrentSize.X());
    retValue.SetY(offset / m_CurrentSize.X());
    return retValue;
}
