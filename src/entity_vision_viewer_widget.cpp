#include "entity_vision_viewer_widget.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
VisionViewerWidget::VisionViewerWidget() :
    m_lockObjectChange(false)
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
    if ( m_lockObjectChange == false )
    {
        m_objEntity = smart_dynamic_pointer_cast<Entity>(obj);
    }
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
        // Lock object change
        ImGui::Checkbox("Lock Object Change: ", &m_lockObjectChange);

        ImGui::Separator();

        // Center vision screen inside imgui widget
        m_cursorPos = ImGui::GetWindowSize();
        m_cursorPos.x = (m_cursorPos.x - SCREEN_WIDTH) * 0.5f;
        m_cursorPos.y = (m_cursorPos.y - SCREEN_HEIGHT) * 0.5f;
        ImGui::SetCursorPos( m_cursorPos );

        EntityVision& vision = m_objEntity->Vision();

        const std::vector<EntityVisionNode::RayData>& visionVector = vision.GetVisionVector();

        const float w = (const float)SCREEN_WIDTH / (const float)visionVector.size();

        m_displayTexture.clear();

        float x_rec = 0.0F;

        for ( volatile const auto& visionData : visionVector )
        {
            float h = SCREEN_HEIGHT - visionData.Distance/SCREEN_HEIGHT * DISTANCE_TO_SCREEN_FACTOR;
            float y_rec = (SCREEN_HEIGHT / 2.0F) - (h / 2.0F);

            m_rectangle.setPosition(x_rec, y_rec);
            m_rectangle.setSize( sf::Vector2f(w, h) );
            m_rectangle.setFillColor( SetColorBrightness( sf::Color(visionData.Fixture), CalculateBrightness(visionData.Distance) ) );
            m_displayTexture.draw(m_rectangle);

            x_rec += w;
        }

        m_displayTexture.display();
        ImGui::Image( m_displayTexture.getTexture() );
    }

    ImGui::PopStyleVar();
    ImGui::End();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
const sf::Color&& VisionViewerWidget::SetColorBrightness(const sf::Color& cl, const float bri)
{
    return std::move(sf::Color(cl.r * bri, cl.g * bri, cl.b * bri));
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
const float VisionViewerWidget::CalculateBrightness(const float distance)
{
    const float ds = 1.0F - distance/SCREEN_HEIGHT * 10U;
    if (ds > 1.0F)
    {
        return 1.0F;
    }
    else if (ds < 0.0F)
    {
        return 0.0F;
    }

    return ds;
}
