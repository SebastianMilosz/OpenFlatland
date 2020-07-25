#include "entity_vision_viewer_widget.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
VisionViewerWidget::VisionViewerWidget() :
    m_lockObjectChange(false),
    m_moveSelectedObject(false),
    m_left(false),
    m_right(false),
    m_up(false),
    m_down(false)
{
    m_renderStates.blendMode = sf::BlendMode(sf::BlendMode::One, sf::BlendMode::OneMinusSrcAlpha);
    m_displayTexture.create(SCREEN_WIDTH, SCREEN_HEIGHT);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void VisionViewerWidget::OnKeyPressed( const sf::Keyboard::Key key )
{
    if ( m_moveSelectedObject )
    {
        if (key == sf::Keyboard::Left)
        {
            m_left = true;
        }
        else if (key == sf::Keyboard::Right)
        {
            m_right = true;
        }
        else if (key == sf::Keyboard::Up)
        {
            m_up = true;
        }
        else if (key == sf::Keyboard::Down)
        {
            m_down = true;
        }

        UpdateSelectedObject();
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void VisionViewerWidget::OnKeyReleased( const sf::Keyboard::Key key )
{
    if ( m_moveSelectedObject )
    {
        if (key == sf::Keyboard::Left)
        {
            m_left = false;
        }
        else if (key == sf::Keyboard::Right)
        {
            m_right = false;
        }
        else if (key == sf::Keyboard::Up)
        {
            m_up = false;
        }
        else if (key == sf::Keyboard::Down)
        {
            m_down = false;
        }

        UpdateSelectedObject();
    }
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

        ImGui::Checkbox("Move Object: ", &m_moveSelectedObject);

        ImGui::Separator();

        // Center vision screen inside imgui widget
        m_cursorPos = ImGui::GetWindowSize();
        m_cursorPos.x = (m_cursorPos.x - SCREEN_WIDTH) * 0.5f;
        m_cursorPos.y = (m_cursorPos.y - SCREEN_HEIGHT) * 0.5f;
        ImGui::SetCursorPos( m_cursorPos );

        EntityVision& vision = m_objEntity->Vision();

        const thrust::host_vector<RayData>& visionVector = vision.GetVisionVector();

        const float w = (const float)SCREEN_WIDTH / (const float)visionVector.size();

        m_displayTexture.clear();

        float x_rec = 0.0F;

        for ( const auto& visionData : visionVector )
        {
            float h = SCREEN_HEIGHT - visionData.Distance/SCREEN_HEIGHT * DISTANCE_TO_SCREEN_FACTOR;
            float y_rec = (SCREEN_HEIGHT / 2.0F) - (h / 2.0F);

            m_rectangle.setPosition(x_rec, y_rec);
            m_rectangle.setSize( sf::Vector2f(w, h) );

            const sf::Color cl(SetColorBrightness( sf::Color(visionData.Fixture), CalculateBrightness(visionData.Distance) ));

            m_rectangle.setFillColor( cl );

            m_displayTexture.draw(m_rectangle, m_renderStates);

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
const sf::Color&& VisionViewerWidget::SetColorBrightness(const sf::Color& cl, const float bri) const
{
    return std::move(sf::Color(cl.r * bri, cl.g * bri, cl.b * bri));
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
const float VisionViewerWidget::CalculateBrightness(const float distance) const
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

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void VisionViewerWidget::UpdateSelectedObject()
{
    if ( smart_ptr_isValid(m_objEntity) )
    {
        EntityMotion& motion = m_objEntity->Motion();

        if ( m_moveSelectedObject )
        {
            if (m_left)
            {
                motion.VelocityRotation = -180.0;
            }
            else if (m_right)
            {
                motion.VelocityRotation = 180.0;
            }
            else
            {
                motion.VelocityRotation = 0.0F;
            }

            if (m_up)
            {
                motion.VelocityForward = 10.0F;
            }
            else if (m_down)
            {
                motion.VelocityForward = -10.0F;
            }
            else
            {
                motion.VelocityForward = 0.0F;
            }
        }
    }
}
