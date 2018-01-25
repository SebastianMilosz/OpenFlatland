#include <iostream>

#include "world.h"
#include "entityfactory.h"
#include "entity.h"

#include <SFML/Graphics.hpp>
#include <TGUI/TGUI.hpp>
#include <Box2D/Box2D.h>

int main()
{
    sf::RenderWindow window(sf::VideoMode(800, 600, 32), "Test");
    tgui::Gui gui(window); // Create the gui and attach it to the window

    auto menu = tgui::MenuBar::create();
    //menu->setRenderer(theme.getRenderer("MenuBar"));
    menu->setSize((float)window.getSize().x, 22.f);
    menu->addMenu("File");
    menu->addMenuItem("Load");
    menu->addMenuItem("Save");
    menu->addMenuItem("Exit");
    menu->addMenu("Edit");
    menu->addMenuItem("Copy");
    menu->addMenuItem("Paste");
    menu->addMenu("Help");
    menu->addMenuItem("About");
    gui.add(menu);

    World           m_World;
    EntityFactory   m_Factory( m_World );

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();

            gui.handleEvent(event); // Pass the event to the widgets
        }

        if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
        {
            int MouseX = sf::Mouse::getPosition(window).x;
            int MouseY = sf::Mouse::getPosition(window).y;
            std::shared_ptr<Entity> entity = m_Factory.Create( MouseX, MouseY, 0 );
        }

        /** Simulate the world */
        m_World.PhysisStep();

        window.clear();

        m_World.Draw( window );

        gui.draw(); // Draw all widgets

        window.display();
    }

    return 0;
}
