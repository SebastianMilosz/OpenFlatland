#include <iostream>

#include <SFML/Graphics.hpp>
#include <Box2D/Box2D.h>

using namespace std;

static const float SCALE = 30.f;

void CreateBox(b2World& World, int MouseX, int MouseY)
{
    b2BodyDef BodyDef;
    BodyDef.position = b2Vec2(MouseX/SCALE, MouseY/SCALE);
    BodyDef.type = b2_dynamicBody;
    b2Body* Body = World.CreateBody(&BodyDef);

    b2PolygonShape Shape;
    Shape.SetAsBox((32.f/2)/SCALE, (32.f/2)/SCALE);
    b2FixtureDef FixtureDef;
    FixtureDef.density = 1.f;
    FixtureDef.friction = 0.7f;
    FixtureDef.shape = &Shape;
    Body->CreateFixture(&FixtureDef);
}

int main()
{
    sf::RenderWindow window(sf::VideoMode(800, 600, 32), "Test");

    b2Vec2 gravity(0.f, 9.8f);
    b2World world(gravity);

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
        {
            int MouseX = sf::Mouse::getPosition(window).x;
            int MouseY = sf::Mouse::getPosition(window).y;
            CreateBox(world, MouseX, MouseY);
        }

        /** Simulate the world */
        world.Step(1/60.f, 8, 3);

        window.clear();

        for(b2Body* BodyIterator = world.GetBodyList(); BodyIterator != 0; BodyIterator = BodyIterator->GetNext())
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

        window.display();
    }

    return 0;
}
