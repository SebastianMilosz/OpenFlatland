#include "information_widget.hpp"
#include "performancelogger.hpp"
#include "performanceapplicationdef.hpp"

#include <chrono>
#include <ctime>
#include <climits>

#include "physics_body.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
InformationWidget::InformationWidget( sf::RenderWindow& window ) :
    m_window( window ),
    m_EntityFactory( NULL ),
    m_curFps( 0 ),
    m_minFps( INT_MAX ),
    m_maxFps( INT_MIN )
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

    m_curFps = GetFps();

    if( m_minFps > m_curFps && m_curFps > 0 )
    {
        m_minFps = m_curFps;
    }
    if( m_maxFps < m_curFps )
    {
        m_maxFps = m_curFps;
    }

    ImGui::Text( "FPS: (%d-%d-%d)", m_minFps, m_curFps, m_maxFps );

    if( NULL != m_EntityFactory )
    {
        ImGui::Text( "Entity Cnt: %d", m_EntityFactory->Count() );
    }

    ImGui::Text( "World Coordinates: (%3.2f, %3.2f)", worldPos.x, worldPos.y );

    ImGui::Text( "Box2d Coordinates: (%3.2f, %3.2f)", worldPos.x/PhysicsBody::sDescriptor::PIXELS_IN_METER, worldPos.y/PhysicsBody::sDescriptor::PIXELS_IN_METER );

    ImGui::Text( PerformanceLogger::GetInstance().PointToString( PERFORMANCE_BOX2D_FULL_PHYSIC_SYM ).c_str() );

    ImGui::Text( PerformanceLogger::GetInstance().PointToString( PERFORMANCE_BOX2D_RAYS_CAST ).c_str() );

    ImGui::Text( PerformanceLogger::GetInstance().PointToString( PERFORMANCE_CALCULATE_NEURONS ).c_str() );

    ImGui::Text( PerformanceLogger::GetInstance().PointToString( PERFORMANCE_RENDER_GRAPHIC ).c_str() );

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
    static int countNbr = 0;
    static auto last = high_resolution_clock::now();
    auto now = high_resolution_clock::now();
    static int fps = 0;

    countNbr++;

    if( duration_cast<milliseconds>(now - last).count() > 1000 )
    {
        fps = countNbr;
        countNbr = 0;
        last = now;
    }

    return fps;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::string InformationWidget::FpsToString()
{
    std::ostringstream ss;

    ss << "(" << m_minFps << "-" << m_curFps << "-" << m_maxFps << ")";

    return std::string( ss.str() );
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
