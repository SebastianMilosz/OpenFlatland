#include "drawable_spiking_neural_network.hpp"

#include "colorize_number.hpp"
#include "font_factory.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
DrawableSpikingNeuralNetwork::DrawableSpikingNeuralNetwork(const std::string& name, ObjectNode* parent) :
    SpikingNeuralNetwork(name, parent)
{
    m_text.setFont(FontFactory::GetFont());
    m_text.setOutlineColor(sf::Color::Red);
    m_text.setCharacterSize(12);
    m_text.setFillColor(sf::Color::Red);

    line[0].color = sf::Color::Blue;
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
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void DrawableSpikingNeuralNetwork::Select(uint32_t x, uint32_t y)
{
    m_selectX = x;
    m_selectY = y;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void DrawableSpikingNeuralNetwork::draw( sf::RenderTarget& target, sf::RenderStates states ) const
{
    const codeframe::Point2D<unsigned int>& poolSize      = CellPoolSize.GetConstValue();
    const thrust::host_vector<float>& synapseLinkVector   = SynapseLink.GetConstValue();
    const thrust::host_vector<float>& synapseWeightVector = SynapseWeight.GetConstValue();

    unsigned int neuronBoxDW = (target.getSize().x - neuronBoxW * (poolSize.X()+1U))/(poolSize.X()+1U);
    unsigned int neuronBoxDH = (target.getSize().y - neuronBoxH * (poolSize.Y()+1U))/(poolSize.Y()+1U);
    unsigned int inW = target.getSize().x / m_dataInput.size();

    unsigned int curX = 0;
    unsigned int curY = 0;

    sf::RectangleShape m_rectangle;
    for(const auto& value : m_dataInput)
    {
        m_rectangle.setOutlineThickness(0U);
        m_rectangle.setFillColor( ColorizeNumber_IronBown<float>(value) );
        m_rectangle.setPosition(curX, curY);
        m_rectangle.setSize( sf::Vector2f(inW, 10) );

        target.draw(m_rectangle, states);

        curX += inW;
    }

    curX = neuronBoxDW;
    curY = neuronBoxDH;

    m_rectangle.setOutlineColor(sf::Color::White);
    m_rectangle.setOutlineThickness(2U);
    m_rectangle.setSize( sf::Vector2f(neuronBoxW, neuronBoxH) );

    for (unsigned int y = 0U; y < poolSize.Y(); y++)
    {
        for (unsigned int x = 0U; x < poolSize.X(); x++)
        {
            unsigned int offset = CoordinateToOffset(x, y);

            float value = IntegrateLevel.GetConstValue()[offset];
            curX = x * (neuronBoxW + neuronBoxDW) + neuronBoxDW;
            curY = y * (neuronBoxH + neuronBoxDH) + neuronBoxDH;

            for (unsigned int n = 0U; n < 100U; n++)
            {
                const float linkValue = synapseLinkVector[offset * GetSynapseSize() + n];
                const float weightValue = synapseWeightVector[offset * GetSynapseSize() + n];

                if (linkValue > 0.5f)
                {
                    volatile unsigned int linkX = OffsetToCoordinate(linkValue).X();
                    volatile unsigned int linkY = OffsetToCoordinate(linkValue).Y();

                    line[0].position.x = curX;
                    line[0].position.y = curY;
                    line[1].position.x = linkX * (neuronBoxW + neuronBoxDW) + neuronBoxDW;
                    line[1].position.y = linkY * (neuronBoxH + neuronBoxDH) + neuronBoxDH;
                    line[1].color = ColorizeNumber_IronBown<float>(weightValue);
                    target.draw(line, 2, sf::Lines);
                }
                else if (linkValue < 0.0f)
                {
                    line[0].position.x = curX;
                    line[0].position.y = curY;
                    line[1].position.x = std::fabs(linkValue) * inW;
                    line[1].position.y = 10U;
                    line[1].color = ColorizeNumber_IronBown<float>(weightValue);
                    target.draw(line, 2, sf::Lines);
                }
                else
                {
                    break;
                }
            }

            sf::Color nodeColor = sf::Color::White;

            if (m_selectX == (int)x && m_selectY == (int)y)
            {
                nodeColor = sf::Color::Red;
            }

            m_rectangle.setOutlineColor(nodeColor);
            m_rectangle.setPosition(curX, curY);
            m_rectangle.setFillColor( ColorizeNumber_IronBown<float>(value) );
            target.draw(m_rectangle, states);

            m_text.setString( utilities::math::IntToStr(offset) );
            m_text.setPosition(curX + neuronBoxW +2, curY + neuronBoxH +2);
            target.draw(m_text);
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::vector<std::tuple<std::string, std::string>> DrawableSpikingNeuralNetwork::GetBlockInfo(uint32_t x, uint32_t y)
{
    std::vector<std::tuple<std::string, std::string>> retInfo;
    retInfo.push_back(std::make_tuple("Test", "Testa"));
    retInfo.push_back(std::make_tuple("Test2", "Testa2"));
    retInfo.push_back(std::make_tuple("Test3", "Testa3"));
    return retInfo;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
uint32_t DrawableSpikingNeuralNetwork::CoordinateToOffset(const uint32_t x, const uint32_t y) const
{
    return m_CurrentSize.Y() * y + x;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
codeframe::Point2D<unsigned int> DrawableSpikingNeuralNetwork::OffsetToCoordinate(const uint32_t offset) const
{
    codeframe::Point2D<unsigned int> retValue;
    retValue.SetX(offset % m_CurrentSize.X());
    retValue.SetY(std::floor(offset / m_CurrentSize.X()));
    return retValue;
}
