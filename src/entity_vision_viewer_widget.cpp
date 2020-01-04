#include "entity_vision_viewer_widget.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
VisionViewerWidget::VisionViewerWidget()
{
    m_displayTexture.create(SCREEN_WIDTH, SCREEN_HEIGHT);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
VisionViewerWidget::~VisionViewerWidget()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void VisionViewerWidget::SetObject( smart_ptr<codeframe::ObjectNode> obj )
{
    m_objEntity = smart_dynamic_pointer_cast<Entity>(obj);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void VisionViewerWidget::Draw( const char* title, bool* p_open )
{
    ImGui::SetNextWindowSize( ImVec2( 430, 450 ), ImGuiCond_FirstUseEver );
    if ( !ImGui::Begin( title, p_open ) )
    {
        ImGui::End();
        return;
    }

    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2(2,2) );

    if ( smart_ptr_isValid(m_objEntity) )
    {
        // Center vision screen inside imgui widget
        m_cursorPos = ImGui::GetWindowSize();
        m_cursorPos.x = (m_cursorPos.x - SCREEN_WIDTH) * 0.5f;
        m_cursorPos.y = (m_cursorPos.y - SCREEN_HEIGHT) * 0.5f;
        ImGui::SetCursorPos( m_cursorPos );

        EntityVision& vision = m_objEntity->Vision();

        const std::vector<float>& distanceVector = vision.GetDistanceVector();

        const unsigned int w = std::ceil((const float)SCREEN_WIDTH / (const float)distanceVector.size());

        m_displayTexture.clear();

        unsigned int x_rec = 0;

        for ( const auto distance : distanceVector )
        {
            float h = SCREEN_HEIGHT - distance/SCREEN_HEIGHT * DISTANCE_TO_SCREEN_FACTOR;
            float y_rec = (SCREEN_HEIGHT / 2.0F) - (h / 2.0F);

            m_rectangle.setPosition(x_rec, y_rec);
            m_rectangle.setSize( sf::Vector2f(w, h) );
            m_displayTexture.draw(m_rectangle);
            x_rec += w;
        }

        m_displayTexture.display();
        ImGui::Image( m_displayTexture.getTexture() );
    }

    ImGui::PopStyleVar();
    ImGui::End();
}
