#include "informationwidget.hpp"

#include <chrono>
#include <ctime>

#include "physicsbody.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
InformationWidget::InformationWidget( sf::RenderWindow& window ) :
    m_window( window ),
    m_EntityFactory( NULL )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
InformationWidget::~InformationWidget()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void InformationWidget::Clear()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void InformationWidget::Draw( const char* title, bool* p_open )
{
    ImGui::Begin("Application Info");

    // get the current mouse position in the window
    sf::Vector2i pixelPos = sf::Mouse::getPosition( m_window );

    // convert it to world coordinates
    sf::Vector2f worldPos = m_window.mapPixelToCoords( pixelPos );

    ImGui::Text( "FPS: %d", GetFps() );

    if( NULL != m_EntityFactory )
    {
        ImGui::Text( "Entity Cnt: %d", m_EntityFactory->Count() );
    }

    ImGui::Text( "World Coordinates: (%3.2f, %3.2f)", worldPos.x, worldPos.y );

    ImGui::Text( "Box2D Coordinates: (%3.2f, %3.2f)", worldPos.x/PhysicsBody::sDescriptor::PIXELS_IN_METER, worldPos.y/PhysicsBody::sDescriptor::PIXELS_IN_METER );

    ImGui::End();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
int InformationWidget::GetFps()
{
    using namespace std::chrono;
    static int count = 0;
    static auto last = high_resolution_clock::now();
    auto now = high_resolution_clock::now();
    static int fps = 0;

    count++;

    if( duration_cast<milliseconds>(now - last).count() > 1000 )
    {
        fps = count;
        count = 0;
        last = now;
    }

    return fps;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void InformationWidget::SetEntityFactory( const EntityFactory& factory )
{
    m_EntityFactory = &factory;
}
