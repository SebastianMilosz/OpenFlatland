#include "ann_viewer_widget.hpp"
#include "colorize_number.hpp"

#include <MathUtilities.h>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
AnnViewerWidget::AnnViewerWidget()
{
    m_renderStates.blendMode = sf::BlendMode(sf::BlendMode::One, sf::BlendMode::OneMinusSrcAlpha);
    m_displayTexture.create(SCREEN_WIDTH, SCREEN_HEIGHT);

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
void AnnViewerWidget::SetObject( smart_ptr<codeframe::ObjectNode> obj )
{
    m_objEntity = smart_dynamic_pointer_cast<Entity>(obj);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void AnnViewerWidget::Draw( const char* title, bool* p_open )
{
    ImGui::SetNextWindowSize( ImVec2( 430, 450 ), ImGuiCond_FirstUseEver );
    if ( !ImGui::Begin( title, p_open ) )
    {
        ImGui::End();
        return;
    }

    ImGui::NextColumn();
    ImGui::PushItemWidth(MENU_LEFT_OFFSET - 30);
        static int selectX = 0;
        ImGui::Text( "X = " ); ImGui::SameLine();
        ImGui::InputInt("##X=", &selectX, 1U);
        static int selectY = 0;
        ImGui::Text( "Y = " ); ImGui::SameLine();
        ImGui::InputInt("##Y=", &selectY, 1U);


    m_displayTexture.clear();

    if ( smart_ptr_isValid(m_objEntity) )
    {
        ArtificialNeuronEngine& engine = m_objEntity->GetEngine();

        const codeframe::Point2D<unsigned int>& poolSize = engine.CellPoolSize.GetConstValue();
        const thrust::host_vector<float>& input = engine.Input.GetConstValue();

        NeuronCellPool& neuronPool = engine.GetPool();

        codeframe::Property< thrust::host_vector<float> >& integrateLevelProperty = neuronPool.IntegrateLevel;
        const thrust::host_vector<float>& integrateLevelVector = integrateLevelProperty.GetConstValue();

        codeframe::Property< thrust::host_vector<float> >& synapseLinkProperty = neuronPool.SynapseLink;
        codeframe::Property< thrust::host_vector<float> >& synapseWeightProperty = neuronPool.SynapseWeight;

        const thrust::host_vector<float>& synapseLinkVector = synapseLinkProperty.GetConstValue();
        const thrust::host_vector<float>& synapseWeightVector = synapseWeightProperty.GetConstValue();

        unsigned int neuronBoxW = 10U;
        unsigned int neuronBoxH = 10U;

        unsigned int neuronBoxDW = (SCREEN_WIDTH  - neuronBoxW * (poolSize.X()+1U))/(poolSize.X()+1U);
        unsigned int neuronBoxDH = (SCREEN_HEIGHT - neuronBoxH * (poolSize.Y()+1U))/(poolSize.Y()+1U);

        unsigned int curX = 0;
        unsigned int curY = 0;

        unsigned int inW = SCREEN_WIDTH / input.size();

        for(const auto& value : input)
        {
            m_rectangle.setOutlineThickness(0U);
            m_rectangle.setFillColor( ColorizeNumber_IronBown<float>(value) );
            m_rectangle.setPosition(curX, curY);
            m_rectangle.setSize( sf::Vector2f(inW, 10) );

            m_displayTexture.draw(m_rectangle, m_renderStates);

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
                unsigned int offset = neuronPool.CoordinateToOffset(x, y);

                float value = integrateLevelVector[offset];
                curX = x * (neuronBoxW + neuronBoxDW) + neuronBoxDW;
                curY = y * (neuronBoxH + neuronBoxDH) + neuronBoxDH;

                for (unsigned int n = 0U; n < 100U; n++)
                {
                    const float linkValue = synapseLinkVector[offset * neuronPool.GetSynapseSize() + n];
                    const float weightValue = synapseWeightVector[offset * neuronPool.GetSynapseSize() + n];

                    if (linkValue > 0.5f)
                    {
                        volatile unsigned int linkX = neuronPool.OffsetToCoordinate(linkValue).X();
                        volatile unsigned int linkY = neuronPool.OffsetToCoordinate(linkValue).Y();

                        line[0].position.x = curX;
                        line[0].position.y = curY;
                        line[1].position.x = linkX * (neuronBoxW + neuronBoxDW) + neuronBoxDW;
                        line[1].position.y = linkY * (neuronBoxH + neuronBoxDH) + neuronBoxDH;
                        line[1].color = ColorizeNumber_IronBown<float>(weightValue);
                        m_displayTexture.draw(line, 2, sf::Lines);
                    }
                    else if (linkValue < 0.0f)
                    {
                        line[0].position.x = curX;
                        line[0].position.y = curY;
                        line[1].position.x = std::fabs(linkValue) * inW;
                        line[1].position.y = 10U;
                        line[1].color = ColorizeNumber_IronBown<float>(weightValue);
                        m_displayTexture.draw(line, 2, sf::Lines);
                    }
                    else
                    {
                        break;
                    }
                }

                sf::Color nodeColor = sf::Color::White;

                if (selectX == x && selectY == y)
                {
                    ImGui::BeginChild("OuterRegion", ImVec2(MENU_LEFT_OFFSET, 300), false);

                    nodeColor = sf::Color::Red;

                    for (unsigned int n = 0U; n < 100U; n++)
                    {
                        const float linkValue = synapseLinkVector[offset * neuronPool.GetSynapseSize() + n];
                        const float weightValue = synapseWeightVector[offset * neuronPool.GetSynapseSize() + n];

                        if (linkValue > 0.5f)
                        {
                            volatile unsigned int linkX = neuronPool.OffsetToCoordinate(linkValue).X();
                            volatile unsigned int linkY = neuronPool.OffsetToCoordinate(linkValue).Y();
                            std::string linkText = utilities::math::FloatToStr(linkValue) +
                            "(" +
                            utilities::math::IntToStr(linkX) +
                            ","  +
                            utilities::math::IntToStr(linkY) +
                            ")(" +
                            utilities::math::FloatToStr(weightValue) + ")";
                            ImGui::Text(linkText.c_str());
                        }
                        else if (linkValue < 0.0f)
                        {
                            std::string linkText = utilities::math::FloatToStr(linkValue) + "(" + utilities::math::IntToStr(std::fabs(linkValue)) + ")(" + utilities::math::FloatToStr(weightValue) + ")";
                            ImGui::Text(linkText.c_str());
                        }
                        else
                        {
                            break;
                        }
                    }

                    ImGui::EndChild();
                }

                m_rectangle.setOutlineColor(nodeColor);
                m_rectangle.setPosition(curX, curY);
                m_rectangle.setFillColor( ColorizeNumber_IronBown<float>(value) );
                m_displayTexture.draw(m_rectangle, m_renderStates);

                m_text.setString( utilities::math::IntToStr(offset) );
                m_text.setPosition(curX + neuronBoxW +2, curY + neuronBoxH +2);
                m_displayTexture.draw(m_text);
            }
        }
    }


    ImGui::PopItemWidth();
    ImGui::NextColumn();

    ImGui::Spacing();

    // Center vision screen inside imgui widget
    m_cursorPos = ImGui::GetWindowSize();
    m_cursorPos.x = (m_cursorPos.x - MENU_LEFT_OFFSET - SCREEN_WIDTH) * 0.5f + MENU_LEFT_OFFSET;
    m_cursorPos.y = (m_cursorPos.y - SCREEN_HEIGHT) * 0.5f;

    if (m_cursorPos.x < MENU_LEFT_OFFSET)
    {
        m_cursorPos.x = MENU_LEFT_OFFSET + 20U;
    }

    ImGui::SetCursorPos( m_cursorPos );
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2(2,2) );

    m_displayTexture.display();
    ImGui::Image( m_displayTexture.getTexture() );

    ImGui::PopStyleVar();

    ImGui::End();
}
