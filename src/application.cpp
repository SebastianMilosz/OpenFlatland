#include "application.hpp"
#include "version.hpp"
#include "performance_logger.hpp"
#include "performance_application_def.hpp"

#include <reference_manager.hpp>
#include <utilities/MathUtilities.h>
#include <utilities/LoggerUtilities.h>
#include <SFML/Graphics.hpp>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
Application::Application( std::string name, sf::RenderWindow& window ) :
    Object( name, NULL ),
    m_zoomAmount( 1.1F ), // zoom by 10%
    m_apiDir( utilities::file::GetExecutablePath() ),
    m_cfgFilePath( m_apiDir + std::string("\\") + std::string( APPLICATION_NAME ) + std::string("_cfg.xml") ),
    m_perFilePath( m_apiDir + std::string("\\") + std::string( APPLICATION_NAME ) + std::string("_per.csv") ),
    m_logFilePath( m_apiDir + std::string("\\") + std::string( APPLICATION_NAME ) + std::string("_log.txt") ),
    m_guiFilePath( m_apiDir + std::string("\\") + std::string( APPLICATION_NAME ) + std::string("_gui.ini") ),
    m_Window ( window ),
    m_Widgets( m_Window, *this, m_guiFilePath ),
    m_World  ( "World", this ),
    m_EntityFactory( "EntityFactory", this ),
    m_ConstElementsFactory( "ConstElementsFactory", this ),
    m_FontFactory( "FontFactory", this ),
    lineCreateState(0)
{
    std::string applicationId  = "0.01";
    /*
                                 std::string( MERCURIAL_AUTHOR ) + std::string(" ") +
                                 std::string( MERCURIAL_DATE_TIME ) + std::string(" ") +
                                 utilities::math::IntToStr( MERCURIAL_REVISION );
    */

    // Logger Setup
    LOGGERINS().LogPath = m_logFilePath;

    LOGGER( LOG_INFO << APPLICATION_NAME << " Start Initializing" );

    PERFORMANCE_INITIALIZE( applicationId );

    m_Window.setTitle( std::string( APPLICATION_NAME ) + std::string(" Rev: ") + applicationId );

    sf::View view( window.getDefaultView() );

    window.setView(view);

    // Connect Signals
    m_EntityFactory.signalEntityAdd.connect( &m_World, &World::AddShell );
    m_EntityFactory.signalContainerSelectionChanged.connect( &m_Widgets.GetPropertyEditorWidget(), &PropertyEditorWidget::SetObject );
    m_EntityFactory.signalContainerSelectionChanged.connect( &m_Widgets.GetAnnViewerWidget(), &AnnViewerWidget::SetObject );
    m_EntityFactory.signalContainerSelectionChanged.connect( &m_Widgets.GetVisionViewerWidget(), &VisionViewerWidget::SetObject );
    m_ConstElementsFactory.signalElementAdd.connect( &m_World, &World::AddConst );

    m_Widgets.GetInformationWidget().SetEntityFactory(  m_EntityFactory  );

    LOGGER( LOG_INFO << APPLICATION_NAME << " Initialized" );

    LOGGERINS().OnMessage.connect(&m_Widgets.GetConsoleWidget(), &ConsoleWidget::OnLogMessage);

    LOGGER( LOG_INFO << APPLICATION_NAME << " Start Processing" );

    PERFORMANCE_ADD( PERFORMANCE_LOAD_XML_FILE,         "Load Xml File" );
    PERFORMANCE_ADD( PERFORMANCE_BOX2D_FULL_PHYSIC_SYM, "Box2d physic symulation" );
    PERFORMANCE_ADD( PERFORMANCE_RENDER_GRAPHIC,        "Render graphic" );
    PERFORMANCE_ADD( PERFORMANCE_CALCULATE_NEURONS,     "Calculate neurons" );
    PERFORMANCE_ADD( PERFORMANCE_SAVE_XML_FILE,         "Save Xml File" );

    PERFORMANCE_ENTER( PERFORMANCE_LOAD_XML_FILE );

    this->Storage().LoadFromFile( m_cfgFilePath );
    this->PulseChanged( true );

    PERFORMANCE_LEAVE( PERFORMANCE_LOAD_XML_FILE );

    // @todo Remove Lua test
    //this->Script().RunString("CF:GetProperty('Application/EntityFactory/Unknown[0].CastRays').Number = 1");
    //this->Script().RunString("CF:GetProperty('Application/EntityFactory/Unknown[1].CastRays').Number = 1");

#if defined(_OPENMP)
    LOGGER( LOG_INFO << "OpenMP Enabled Version:" << _OPENMP );
#else
    LOGGER( LOG_INFO << "OpenMP Disabled" );
#endif
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
        LOGGER( LOG_INFO << "Application closed event" );

        codeframe::ReferenceManager::LogUnresolvedReferences();

        // Add fps note to performance log
        std::string note  = std::string("FPS="  ) + m_Widgets.GetInformationWidget().FpsToString();
                    note += std::string(", CNT=") + utilities::math::IntToStr( m_EntityFactory.Count() );
        PerformanceLogger::GetInstance().AddNote( note );

        PERFORMANCE_ENTER( PERFORMANCE_SAVE_XML_FILE );
        this->Storage().SaveToFile( m_cfgFilePath );
        PERFORMANCE_LEAVE( PERFORMANCE_SAVE_XML_FILE );

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

    else if ( event.type == sf::Event::KeyPressed )
    {
        m_Widgets.GetVisionViewerWidget().OnKeyPressed(event.key.code);
    }

    else if ( event.type == sf::Event::KeyReleased )
    {
        m_Widgets.GetVisionViewerWidget().OnKeyReleased(event.key.code);
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

    PERFORMANCE_ENTER( PERFORMANCE_BOX2D_FULL_PHYSIC_SYM );

    /** Simulate the world */
    m_World.PhysisStep( m_Window );

    PERFORMANCE_LEAVE( PERFORMANCE_BOX2D_FULL_PHYSIC_SYM );

    PERFORMANCE_ENTER( PERFORMANCE_CALCULATE_NEURONS );

    m_EntityFactory.CalculateNeuralNetworks();

    PERFORMANCE_LEAVE( PERFORMANCE_CALCULATE_NEURONS );

    PERFORMANCE_ENTER( PERFORMANCE_RENDER_GRAPHIC );

    m_World.synchronize();
    m_World.draw( m_Window, sf::RenderStates::Default );

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

    PERFORMANCE_LEAVE( PERFORMANCE_RENDER_GRAPHIC );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void Application::ZoomViewAt( sf::Vector2i pixel, sf::RenderWindow& window, const float zoom )
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
