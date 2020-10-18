#include "ann_viewer_widget.hpp"
#include "colorize_number.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
AnnViewerWidget::AnnViewerWidget()
{
    m_renderStates.blendMode = sf::BlendMode(sf::BlendMode::One, sf::BlendMode::OneMinusSrcAlpha);
    m_displayTexture.create(SCREEN_WIDTH, SCREEN_HEIGHT);
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

    // Center vision screen inside imgui widget
    m_cursorPos = ImGui::GetWindowSize();
    m_cursorPos.x = (m_cursorPos.x - SCREEN_WIDTH) * 0.5f;
    m_cursorPos.y = (m_cursorPos.y - SCREEN_HEIGHT) * 0.5f;
    ImGui::SetCursorPos( m_cursorPos );

    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2(2,2) );

    m_displayTexture.clear();

    if ( smart_ptr_isValid(m_objEntity) )
    {
        ArtificialNeuronEngine& engine = m_objEntity->GetEngine();

        const codeframe::Point2D<unsigned int>& poolSize = engine.CellPoolSize.GetConstValue();
        const thrust::host_vector<float>& input = engine.Input.GetConstValue();

        NeuronCellPool& neuronPool = engine.GetPool();

        codeframe::Property< thrust::host_vector<float> >& integrateLevelProperty = neuronPool.IntegrateLevel;
        const thrust::host_vector<float> integrateLevelVector = integrateLevelProperty.GetConstValue();

        unsigned int neuronBoxW = 3U;
        unsigned int neuronBoxH = 3U;

        unsigned int neuronBoxDW = (SCREEN_WIDTH  - (3U * poolSize.X())) / (poolSize.X() + 1U);
        unsigned int neuronBoxDH = (SCREEN_HEIGHT - (3U * poolSize.Y())) / (poolSize.Y() + 1U);

        unsigned int curX = 0;
        unsigned int curY = 0;

        unsigned int inW = SCREEN_WIDTH / input.size();

        for(const auto& value : input)
        {
            (void)value;
            m_rectangle.setOutlineColor(sf::Color::White);
            m_rectangle.setOutlineThickness(0U);
            m_rectangle.setFillColor( ColorizeNumber_IronBown<float>(value) );
            m_rectangle.setPosition(curX, curY);
            m_rectangle.setSize( sf::Vector2f(inW, 10) );

            m_displayTexture.draw(m_rectangle, m_renderStates);

            curX += inW;
        }

        curX = neuronBoxDW;
        curY = neuronBoxDH;

        for(const auto& value : integrateLevelVector)
        {
            m_rectangle.setPosition(curX, curY);
            m_rectangle.setSize( sf::Vector2f(neuronBoxW, neuronBoxH) );
            m_rectangle.setOutlineColor(sf::Color::White);
            m_rectangle.setOutlineThickness(2U);
            m_rectangle.setFillColor( ColorizeNumber_IronBown<float>(value) );

            m_displayTexture.draw(m_rectangle, m_renderStates);

            curX += (neuronBoxW + neuronBoxDW);
            if (curX > (SCREEN_WIDTH - (neuronBoxH + neuronBoxDH)))
            {
                curX = neuronBoxDW;
                curY += (neuronBoxH + neuronBoxDH);
            }
        }
    }

    m_displayTexture.display();
    ImGui::Image( m_displayTexture.getTexture() );

    ImGui::PopStyleVar();
    ImGui::End();
}
