#include <iostream>

#include "world.h"
#include "entityfactory.h"
#include "guiwidgetslayer.h"
#include "entity.h"

#include <SFML/Graphics.hpp>
#include <TGUI/TGUI.hpp>
#include <Box2D/Box2D.h>

int main()
{
    sf::RenderWindow window(sf::VideoMode(800, 600, 32), "Test");

    GUIWidgetsLayer m_Widgets( window );
    World           m_World;
    EntityFactory   m_Factory( m_World );

    while (window.isOpen())
    {
        bool guiConsumeEvent = false;

        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();

            guiConsumeEvent = m_Widgets.HandleEvent(event);
        }

        if( guiConsumeEvent == false )
        {
            if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
            {
                int MouseX = sf::Mouse::getPosition(window).x;
                int MouseY = sf::Mouse::getPosition(window).y;
                std::shared_ptr<Entity> entity = m_Factory.Create( MouseX, MouseY, 0 );
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
