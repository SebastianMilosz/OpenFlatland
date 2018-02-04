#include <iostream>

#include "world.h"
#include "entityfactory.h"
#include "guiwidgetslayer.h"
#include "entity.h"

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
    const float zoomAmount{ 1.1f }; // zoom by 10%

    sf::RenderWindow window(sf::VideoMode(800, 600, 32), "Life Simulator");

    sf::View view( window.getDefaultView() );

    window.setView(view);

    GUIWidgetsLayer m_Widgets( window );
    World           m_World;
    EntityFactory   m_Factory( m_World );

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
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

            else if (event.type == sf::Event::MouseWheelScrolled)
			{
				if (event.mouseWheelScroll.delta > 0)
					zoomViewAt({ event.mouseWheelScroll.x, event.mouseWheelScroll.y }, window, (1.f / zoomAmount));
				else if (event.mouseWheelScroll.delta < 0)
					zoomViewAt({ event.mouseWheelScroll.x, event.mouseWheelScroll.y }, window, zoomAmount);
			}

            m_Widgets.HandleEvent(event);
        }

        if( m_Widgets.MouseOnGui() == false )
        {
            if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
            {
                // get the current mouse position in the window
                sf::Vector2i pixelPos = sf::Mouse::getPosition( window );

                // convert it to world coordinates
                sf::Vector2f worldPos = window.mapPixelToCoords( pixelPos );

                int MouseX = worldPos.x;
                int MouseY = worldPos.y;

                if( m_Widgets.GetMouseModeId() == GUIWidgetsLayer::MOUSE_MODE_ADD_ENTITY )
                {
                    std::shared_ptr<Entity> entity = m_Factory.Create( MouseX, MouseY, 0 );
                }
                else if( m_Widgets.GetMouseModeId() == GUIWidgetsLayer::MOUSE_MODE_SEL_ENTITY )
                {
                    m_World.MouseDown( MouseX, MouseY );
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

    return 0;
}
