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

    m_displayTexture.clear();

    if ( smart_ptr_isValid(m_objEntity) )
    {
        ArtificialNeuronEngine& engine = m_objEntity->GetEngine();
        DrawableSpikingNeuralNetwork& neuronPool = dynamic_cast<DrawableSpikingNeuralNetwork&>(engine.GetPool());

        neuronPool.Select(selectX, selectY);
        m_displayTexture.draw(neuronPool, m_renderStates);
    }

    m_displayTexture.display();
    ImGui::Image( m_displayTexture.getTexture() );

    ImGui::PopStyleVar();

    ImGui::End();
}
