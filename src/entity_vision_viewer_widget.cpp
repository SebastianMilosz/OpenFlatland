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

        const std::vector<EntityVisionNode::RayData>& visionVector = vision.GetVisionVector();

        const uint32_t w = std::ceil((const float)SCREEN_WIDTH / (const float)visionVector.size());

        m_displayTexture.clear();

        uint32_t cl_r = 255U;
        uint32_t cl_g = 255U;
        uint32_t cl_b = 255U;
        uint32_t x_rec = 0;

        for ( const auto& visionData : visionVector )
        {
            float ds = visionData.Distance/SCREEN_HEIGHT;
            cl_r = (255U - 255U * ds * 10U);
            cl_g = (255U - 255U * ds * 10U);
            cl_b = (255U - 255U * ds * 10U);
            if ( cl_r < 255 && cl_g < 255 && cl_b < 255 )
            {
                float h = SCREEN_HEIGHT - visionData.Distance/SCREEN_HEIGHT * DISTANCE_TO_SCREEN_FACTOR;
                float y_rec = (SCREEN_HEIGHT / 2.0F) - (h / 2.0F);

                m_rectangle.setPosition(x_rec, y_rec);
                m_rectangle.setSize( sf::Vector2f(w, h) );
                m_rectangle.setFillColor( sf::Color(cl_r, cl_g, cl_b) );
                m_displayTexture.draw(m_rectangle);
            }
            x_rec += w;
        }

        m_displayTexture.display();
        ImGui::Image( m_displayTexture.getTexture() );
    }

    ImGui::PopStyleVar();
    ImGui::End();
}