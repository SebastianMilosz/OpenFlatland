#include <iostream>

#include "version.h"
#include "world.h"
#include "entityfactory.h"
#include "guiwidgetslayer.h"
#include "entity.h"
#include "logwidget.h"

#include <utilities/LoggerUtilities.h>
#include <utilities/FileUtilities.h>
#include <serializable.h>
#include <cpgf/gcallbacklist.h>
#include <SFML/Graphics.hpp>
#include <Box2D/Box2D.h>

void zoomViewAt( sf::Vector2i pixel, sf::RenderWindow& window, float zoom )
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

int main()
{
    codeframe::CODEFRAME_TYPES_INITIALIZE();

    // Logger Setup
    std::string apiDir = utilities::file::GetExecutablePath();
    std::string logFilePath = apiDir + std::string("\\") + std::string( APPLICATION_NAME ) + std::string("_log.txt");
    std::string cfgFilePath = apiDir + std::string("\\") + std::string( APPLICATION_NAME ) + std::string("_cfg.xml");
    LOGGERINS().LogPath = logFilePath;

    LOGGER( LOG_INFO << APPLICATION_NAME << " Start Initializing" );

    const float zoomAmount{ 1.1f }; // zoom by 10%

    sf::RenderWindow window(sf::VideoMode(800, 600, 32), APPLICATION_NAME );

    sf::View view( window.getDefaultView() );

    window.setView(view);

    GUIWidgetsLayer m_Widgets( window );
    World           m_World;
    EntityFactory   m_Factory( m_World );

    m_Factory.LoadFromFile( cfgFilePath );

    LOGGER( LOG_INFO << APPLICATION_NAME << " Initialized" );

    LOGGERINS().OnMessage.connect(&m_Widgets.GetLogWidget(), &LogWidget::OnLogMessage);

    LOGGER( LOG_INFO << APPLICATION_NAME << " Start Processing" );

    while (window.isOpen())
    {
        // get the current mouse position in the window
        sf::Vector2i pixelPos = sf::Mouse::getPosition( window );

        // convert it to world coordinates
        sf::Vector2f worldPos = window.mapPixelToCoords( pixelPos );

        sf::Event event;
        while ( window.pollEvent(event) )
        {
            if (event.type == sf::Event::Closed)
                window.close();

            // catch the resize events
            if (event.type == sf::Event::Resized)
            {
                // update the view to the new size of the window
                sf::FloatRect visibleArea(0, 0, event.size.width, event.size.height);
                window.setView(sf::View(visibleArea));
            }

            else if ( event.type == sf::Event::MouseWheelScrolled )
			{
				if (event.mouseWheelScroll.delta > 0)
					zoomViewAt({ event.mouseWheelScroll.x, event.mouseWheelScroll.y }, window, (1.f / zoomAmount));
				else if (event.mouseWheelScroll.delta < 0)
					zoomViewAt({ event.mouseWheelScroll.x, event.mouseWheelScroll.y }, window, zoomAmount);
			}

			else if ( event.type == sf::Event::MouseButtonReleased )
            {
                m_World.MouseUp( worldPos.x, worldPos.y );
            }

            else if ( event.type == sf::Event::MouseMoved )
            {
                m_World.MouseMove( worldPos.x, worldPos.y );
            }

            m_Widgets.HandleEvent(event);
        }

        if ( m_Widgets.MouseOnGui() == false )
        {
            if ( sf::Mouse::isButtonPressed( sf::Mouse::Left ) )
            {
                if ( m_Widgets.GetMouseModeId() == GUIWidgetsLayer::MOUSE_MODE_ADD_ENTITY )
                {
                    m_World.AddShell( std::dynamic_pointer_cast<EntityShell>( m_Factory.Create( worldPos.x, worldPos.y, 0 ) ) );
                }
                else if ( m_Widgets.GetMouseModeId() == GUIWidgetsLayer::MOUSE_MODE_SEL_ENTITY )
                {
                    m_World.MouseDown( worldPos.x, worldPos.y );
                }
            }
        }

        /** Simulate the world */
        m_World.PhysisStep();

        window.clear();

        m_World.Draw( window );

        m_Widgets.Draw();

        window.display();
    }

    m_Factory.SaveToFile( cfgFilePath );

    return 0;
}
