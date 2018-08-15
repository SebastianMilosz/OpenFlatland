#include "application.hpp"
#include "version.h"

#include <utilities/LoggerUtilities.h>
#include <cpgf/gcallbacklist.h>
#include <SFML/Graphics.hpp>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
Application::Application( std::string name, sf::RenderWindow& window ) :
    cSerializable( name, NULL ),
    m_zoomAmount( 1.1F ), // zoom by 10%
    m_cfgFilePath(""),
    m_Window ( window ),
    m_Widgets( m_Window ),
    m_World  ( "World", this ),
    m_EntityFactory( "EntityFactory", this ),
    m_ConstElementsFactory( "ConstElementsFactory", this ),
    m_FontFactory( "FontFactory", this ),
    lineCreateState(0)
{
    // Logger Setup
    std::string apiDir = utilities::file::GetExecutablePath();
    std::string logFilePath = apiDir + std::string("\\") + std::string( APPLICATION_NAME ) + std::string("_log.txt");
    m_cfgFilePath = apiDir + std::string("\\") + std::string( APPLICATION_NAME ) + std::string("_cfg.xml");
    LOGGERINS().LogPath = logFilePath;

    LOGGER( LOG_INFO << APPLICATION_NAME << " Start Initializing" );

    m_Window.setTitle( APPLICATION_NAME );

    sf::View view( window.getDefaultView() );

    window.setView(view);

    // Connect Signals
    m_EntityFactory       .signalEntityAdd .connect( &m_World, &World::AddShell );
    m_ConstElementsFactory.signalElementAdd.connect( &m_World, &World::AddConst );

    LOGGER( LOG_INFO << APPLICATION_NAME << " Initialized" );

    LOGGERINS().OnMessage.connect(&m_Widgets.GetLogWidget(), &LogWidget::OnLogMessage);

    LOGGER( LOG_INFO << APPLICATION_NAME << " Start Processing" );

    this->LoadFromFile( m_cfgFilePath );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
Application::~Application()
{
    this->SaveToFile( m_cfgFilePath );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void Application::ProcesseEvents( sf::Event& event )
{
    // get the current mouse position in the window
    sf::Vector2i pixelPos = sf::Mouse::getPosition( m_Window );

    // convert it to world coordinates
    sf::Vector2f worldPos = m_Window.mapPixelToCoords( pixelPos );

    // catch window close event
    if ( event.type == sf::Event::Closed )
    {
        m_Window.close();
    }

    // catch the resize events
    if ( event.type == sf::Event::Resized )
    {
        // update the view to the new size of the window
        sf::FloatRect visibleArea( 0, 0, event.size.width, event.size.height );
        m_Window.setView( sf::View( visibleArea ) );
    }

    // catch MouseWheel event
    else if ( event.type == sf::Event::MouseWheelScrolled )
    {
        if (event.mouseWheelScroll.delta > 0)
        {
            ZoomViewAt({ event.mouseWheelScroll.x, event.mouseWheelScroll.y }, m_Window, (1.f / m_zoomAmount));
        }
        else if (event.mouseWheelScroll.delta < 0)
        {
            ZoomViewAt({ event.mouseWheelScroll.x, event.mouseWheelScroll.y }, m_Window, m_zoomAmount);
        }
    }

    else if ( event.type == sf::Event::MouseButtonReleased )
    {
        m_World.MouseUp( worldPos.x, worldPos.y );

        if( m_Widgets.GetMouseModeId() == GUIWidgetsLayer::MOUSE_MODE_ADD_LINE )
        {
            if( lineCreateState == 1 )
            {
                lineCreateState = 2;
                startPoint = worldPos;
            }
        }
    }

    else if ( event.type == sf::Event::MouseMoved )
    {
        m_World.MouseMove( worldPos.x, worldPos.y );

        if( m_Widgets.GetMouseModeId() == GUIWidgetsLayer::MOUSE_MODE_ADD_LINE )
        {
            if( lineCreateState == 2 )
            {
                endPoint = worldPos;
            }
        }
    }

    m_Widgets.HandleEvent(event);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void Application::ProcesseLogic( void )
{
    // get the current mouse position in the window
    sf::Vector2i pixelPos = sf::Mouse::getPosition( m_Window );

    // convert it to world coordinates
    sf::Vector2f worldPos = m_Window.mapPixelToCoords( pixelPos );

    if ( m_Widgets.MouseOnGui() == false )
    {
        if ( sf::Mouse::isButtonPressed( sf::Mouse::Left ) )
        {
            if ( m_Widgets.GetMouseModeId() == GUIWidgetsLayer::MOUSE_MODE_ADD_ENTITY )
            {
                m_EntityFactory.Create( worldPos.x, worldPos.y, 0 );
            }
            else if ( m_Widgets.GetMouseModeId() == GUIWidgetsLayer::MOUSE_MODE_SEL_ENTITY )
            {
                m_World.MouseDown( worldPos.x, worldPos.y );
            }
            else if( m_Widgets.GetMouseModeId() == GUIWidgetsLayer::MOUSE_MODE_ADD_LINE )
            {
                if( lineCreateState == 0 )
                {
                    startPoint = worldPos;
                    lineCreateState = 1;
                }
                else if( lineCreateState == 2 )
                {
                    lineCreateState = 0;

                    // Create solid line
                    m_ConstElementsFactory.CreateLine( codeframe::Point2D( startPoint.x, startPoint.y ), codeframe::Point2D( endPoint.x, endPoint.y ) );
                }
            }
            else
            {
                startPoint = worldPos;
                endPoint = worldPos;
            }
        }
    }

    /** Simulate the world */
    m_World.PhysisStep();

    m_Window.clear();

    m_World.Draw( m_Window );

    if( m_Widgets.GetMouseModeId() == GUIWidgetsLayer::MOUSE_MODE_ADD_LINE )
    {
        if( lineCreateState == 2 )
        {
            sf::Vertex line[] =
            {
                sf::Vertex( startPoint ),
                sf::Vertex( endPoint )
            };

            m_Window.draw(line, 2, sf::Lines);
        }
    }

    m_Widgets.Draw();

    m_Window.display();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void Application::ZoomViewAt( sf::Vector2i pixel, sf::RenderWindow& window, float zoom )
{
	const sf::Vector2f beforeCoord{ window.mapPixelToCoords(pixel) };
	sf::View view{ window.getView() };
	view.zoom(zoom);
	window.setView(view);
	const sf::Vector2f afterCoord{ window.mapPixelToCoords(pixel) };
	const sf::Vector2f offsetCoords{ beforeCoord - afterCoord };
	view.move(offsetCoords);
	window.setView(view);
}
