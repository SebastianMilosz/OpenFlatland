#include "drawable_spiking_neural_network.hpp"

#include "colorize_number.hpp"
#include "font_factory.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
DrawableSpikingNeuralNetwork::DrawableSpikingNeuralNetwork(const std::string& name, ObjectNode* parent) :
    SpikingNeuralNetwork(name, parent),
    m_text(FontFactory::GetFont())
{
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
    m_dataOutput = dataOutput;
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

    // If no data do nothing
    if (m_dataInput.size() == 0U)
    {
        return;
    }

    unsigned int neuronBoxDW = (target.getSize().x - neuronBoxW * (poolSize.X()+1U))/(poolSize.X()+1U);
    unsigned int neuronBoxDH = (target.getSize().y - neuronBoxH * (poolSize.Y()+1U))/(poolSize.Y()+1U);
    float inW = target.getSize().x / m_dataInput.size();

    float curX = 0;
    float curY = 0;

    sf::RectangleShape m_rectangle;
    // Draw imput data
    for(const auto& value : m_dataInput)
    {
        m_rectangle.setOutlineThickness(0U);
        m_rectangle.setFillColor( ColorizeNumber_IronBown<float>(value) );
        m_rectangle.setPosition({curX, curY});
        m_rectangle.setSize( sf::Vector2f(inW, 10) );

        target.draw(m_rectangle, states);

        curX += inW;
    }

    if (m_dataOutput.size() > 0)
    {
        curX = 0;
        curY = target.getSize().y - 10U;
        float outW  = target.getSize().x / m_dataOutput.size();
        m_rectangle.setOutlineColor(sf::Color::Blue);

        // Draw output data
        for (const auto& value : m_dataOutput)
        {
            m_rectangle.setOutlineThickness(1U);
            m_rectangle.setFillColor(ColorizeNumber_IronBown<float>(value));
            m_rectangle.setPosition({curX, curY});
            m_rectangle.setSize({outW, 10.0});

            target.draw(m_rectangle, states);

            curX += outW;
        }
    }

    curX = neuronBoxDW;
    curY = neuronBoxDH;

    m_rectangle.setOutlineColor(sf::Color::White);
    m_rectangle.setOutlineThickness(2U);
    m_rectangle.setSize(sf::Vector2f(neuronBoxW, neuronBoxH));

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

                if (weightValue > 0.0f)
                {
                    if (linkValue > 0.5f)
                    {
                        volatile unsigned int linkX = OffsetToCoordinate(linkValue).X();
                        volatile unsigned int linkY = OffsetToCoordinate(linkValue).Y();

                        line[0].position.x = curX;
                        line[0].position.y = curY;
                        line[1].position.x = linkX * (neuronBoxW + neuronBoxDW) + neuronBoxDW;
                        line[1].position.y = linkY * (neuronBoxH + neuronBoxDH) + neuronBoxDH;
                        line[1].color = ColorizeNumber_IronBown<float>(weightValue);
                        target.draw(line, 2, sf::PrimitiveType::Lines);
                    }
                    else if (linkValue < 0.0f)
                    {
                        line[0].position.x = curX;
                        line[0].position.y = curY;
                        line[1].position.x = std::fabs(linkValue) * inW;
                        line[1].position.y = 10U;
                        line[1].color = ColorizeNumber_IronBown<float>(weightValue);
                        target.draw(line, 2, sf::PrimitiveType::Lines);
                    }
                    else
                    {
                        break;
                    }
                }
            }

            sf::Color nodeColor = sf::Color::White;

            if (m_selectX == (int)x && m_selectY == (int)y)
            {
                nodeColor = sf::Color::Red;
            }

            m_rectangle.setOutlineColor(nodeColor);
            m_rectangle.setPosition({curX, curY});
            m_rectangle.setFillColor( ColorizeNumber_IronBown<float>(value) );
            target.draw(m_rectangle, states);

            m_text.setString( utilities::math::IntToStr(offset) );
            m_text.setPosition({curX + neuronBoxW +2, curY + neuronBoxH +2});
            target.draw(m_text);
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
DrawableSpikingNeuralNetwork::BlockInfo DrawableSpikingNeuralNetwork::GetBlockInfo(uint32_t x, uint32_t y)
{
    BlockInfo retInfo;

    if (m_CurrentSize.X() == 0 && m_CurrentSize.Y() == 0)
    {
        return retInfo;
    }

    unsigned int offset = CoordinateToOffset(x, y);

    std::string linkText2 = "(" + utilities::math::IntToHex(m_Output[offset]) + "," +
                                  utilities::math::IntToStr(m_IntegrateInterval[offset]) +
                            ")";
    std::string linkText = "(" +
            utilities::math::IntToStr(x) +
            ","  +
            utilities::math::IntToStr(y) + ")";

    retInfo.push_back(std::make_tuple(linkText, linkText2));

    linkText2 = "(" + utilities::math::FloatToStr(m_IntegrateLevel[offset], "%2.3f") + "," +
                      utilities::math::FloatToStr(m_IntegrateThreshold[offset], "%2.3f") +
                ")";
    linkText = "(" +
            utilities::math::IntToStr(x) +
            ","  +
            utilities::math::IntToStr(y) + ")";

    retInfo.push_back(std::make_tuple(linkText, linkText2));
    retInfo.push_back(std::make_tuple("------", "------"));

    for (unsigned int n = 0U; n < 100U; n++)
    {
        const float linkValue = SynapseLink.GetConstValue()[offset * GetSynapseSize() + n];
        const float weightValue = SynapseWeight.GetConstValue()[offset * GetSynapseSize() + n];

        if (weightValue > 0.0f)
        {
            if (linkValue > 0.0f)
            {
                volatile unsigned int linkX = OffsetToCoordinate(linkValue).X();
                volatile unsigned int linkY = OffsetToCoordinate(linkValue).Y();

                linkText = "(" +
                utilities::math::IntToStr(linkX) +
                ","  +
                utilities::math::IntToStr(linkY) +
                ")(" +
                utilities::math::FloatToStr(weightValue, "%2.3f") +
                ")";

                double intpart;
                uint8_t bitPos = 64U * modf(linkValue , &intpart);
                uint64_t outVal = m_Output[intpart];
                float value = ((outVal & (1U<<bitPos)) > 0.0f);
                linkText2 = "(" + utilities::math::FloatToStr(value, "%2.3f") + ") :";
            }
            else if (linkValue < 0.0f)
            {
                linkText = "(" +
                utilities::math::IntToStr(std::fabs(linkValue)) +
                ")(" + utilities::math::FloatToStr(weightValue, "%2.3f") + ")";

                unsigned int inPos = std::fabs(linkValue);
                float value = m_dataInput[inPos];
                linkText2 = "(" + utilities::math::FloatToStr(value, "%2.3f") + ") :";
            }
            else
            {
                break;
            }

            retInfo.push_back(std::make_tuple(linkText2, linkText));
        }
    }

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
