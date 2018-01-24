#include <iostream>

#include "entityfactory.h"
#include "entity.h"

#include <SFML/Graphics.hpp>
#include <TGUI/TGUI.hpp>
#include <Box2D/Box2D.h>

using namespace std;

static const float SCALE = 30.f;

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

    b2Vec2          m_Gravity( 0.f, 0.f );
    b2World         m_World( m_Gravity );
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
        m_World.Step(1/60.f, 8, 3);

        window.clear();

        for(b2Body* BodyIterator = m_World.GetBodyList(); BodyIterator != 0; BodyIterator = BodyIterator->GetNext())
        {
            if (BodyIterator->GetType() == b2_dynamicBody)
            {
                sf::CircleShape circle;
                circle.setRadius(10);
                circle.setOutlineColor(sf::Color::Red);
                circle.setOutlineThickness(5);
                circle.setOrigin(16.f, 16.f);
                circle.setPosition(SCALE * BodyIterator->GetPosition().x, SCALE * BodyIterator->GetPosition().y);
                circle.setRotation(BodyIterator->GetAngle() * 180/b2_pi);
                window.draw(circle);
            }
            else
            {
                sf::Sprite GroundSprite;
                //GroundSprite.SetTexture(GroundTexture);
                GroundSprite.setOrigin(400.f, 8.f);
                GroundSprite.setPosition(BodyIterator->GetPosition().x * SCALE, BodyIterator->GetPosition().y * SCALE);
                GroundSprite.setRotation(180/b2_pi * BodyIterator->GetAngle());
                window.draw(GroundSprite);
            }
        }

        gui.draw(); // Draw all widgets

        window.display();
    }

    return 0;
}
