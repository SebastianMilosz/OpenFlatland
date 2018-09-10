#include "application.hpp"
#include "version.hpp"
#include "mercurialinfo.hpp"
#include "performancelogger.hpp"

#include <utilities/MathUtilities.h>
#include <utilities/LoggerUtilities.h>
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
    std::string applicationId  = std::string( MERCURIAL_AUTHOR ) + std::string(" ") +
                                 std::string( MERCURIAL_DATE_TIME ) + std::string(" ") +
                                 utilities::math::IntToStr( MERCURIAL_REVISION );

    // Logger Setup
    std::string apiDir = utilities::file::GetExecutablePath();
    std::string logFilePath = apiDir + std::string("\\") + std::string( APPLICATION_NAME ) + std::string("_log.txt");
    m_cfgFilePath = apiDir + std::string("\\") + std::string( APPLICATION_NAME ) + std::string("_cfg.xml");
    m_perFilePath = apiDir + std::string("\\") + std::string( APPLICATION_NAME ) + std::string("_per.csv");
    LOGGERINS().LogPath = logFilePath;

    LOGGER( LOG_INFO << APPLICATION_NAME << " Start Initializing" );

    PERFORMANCE_INITIALIZE( applicationId );

    m_Window.setTitle( std::string( APPLICATION_NAME ) + std::string(" Rev: ") + utilities::math::IntToStr( MERCURIAL_REVISION ) );

    sf::View view( window.getDefaultView() );

    window.setView(view);

    // Connect Signals
    m_EntityFactory       .signalEntityAdd .connect( &m_World, &World::AddShell );
    m_EntityFactory       .signalContainerSelectionChanged.connect( &m_Widgets.GetPropertyEditorWidget(), &PropertyEditorWidget::SetObject );
    m_EntityFactory       .signalContainerSelectionChanged.connect( &m_Widgets.GetAnnViewerWidget(), &AnnViewerWidget::SetObject );
    m_ConstElementsFactory.signalElementAdd.connect( &m_World, &World::AddConst );

    m_Widgets.GetInformationWidget().SetEntityFactory(  m_EntityFactory  );

    LOGGER( LOG_INFO << APPLICATION_NAME << " Initialized" );

    LOGGERINS().OnMessage.connect(&m_Widgets.GetLogWidget(), &LogWidget::OnLogMessage);

    LOGGER( LOG_INFO << APPLICATION_NAME << " Start Processing" );

    PERFORMANCE_ADD( 1, "Load Xml File" );
    PERFORMANCE_ADD( 2, "Box2d physic symulation" );
    PERFORMANCE_ADD( 3, "Render graphic" );

    PERFORMANCE_ENTER( 1 );

    this->LoadFromFile( m_cfgFilePath );

    PERFORMANCE_LEAVE( 1 );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
Application::~Application()
{
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
        this->SaveToFile( m_cfgFilePath );
        PERFORMANCE_SAVE( m_perFilePath );
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
                    m_ConstElementsFactory.CreateLine( codeframe::Point2D<int>( startPoint ), codeframe::Point2D<int>( endPoint ) );
                }
            }
            else
            {
                startPoint = worldPos;
                endPoint = worldPos;
            }
        }
    }

    m_Window.clear();

    PERFORMANCE_ENTER( 2 );

    /** Simulate the world */
    m_World.PhysisStep( m_Window );

    PERFORMANCE_LEAVE( 2 );

    PERFORMANCE_ENTER( 3 );

    m_World.Draw( m_Window );

    PERFORMANCE_LEAVE( 3 );

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
