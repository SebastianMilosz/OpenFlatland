#include "entity_vision_viewer_widget.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
VisionViewerWidget::VisionViewerWidget()
{
    //ctor
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

    m_displayImage.create(640, 480, sf::Color::Black);
    m_displayTexture.create(640, 480);
    m_displayTexture.update(m_displayImage);
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

    ImVec2 vec;
    vec.x = (ImGui::GetWindowSize().x - 640) * 0.5f;
    vec.y = (ImGui::GetWindowSize().y - 480) * 0.5f;

    ImGui::SetCursorPos( vec );



    m_displayTexture.update(m_displayImage);

    ImGui::Image( m_displayTexture, sf::Color::White, sf::Color::White );

    ImGui::PopStyleVar();
    ImGui::End();
}
