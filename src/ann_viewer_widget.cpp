#include "ann_viewer_widget.hpp"

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
        NeuronCellPool& neuronPool = engine.GetPool();

        codeframe::Property< thrust::host_vector<float> >& integrateLevelProperty = neuronPool.IntegrateLevel;
        const thrust::host_vector<float> integrateLevelVector = integrateLevelProperty.GetConstValue();

        for(const auto& value : integrateLevelVector)
        {
            const sf::Color cl(value);

            m_rectangle.setFillColor( cl );

            m_displayTexture.draw(m_rectangle, m_renderStates);
        }
    }

    m_displayTexture.display();
    ImGui::Image( m_displayTexture.getTexture() );

    ImGui::PopStyleVar();
    ImGui::End();
}
