#include "world.h"

static const float SCALE = 30.f;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
World::World() :
    m_Gravity( 0.f, 0.f ),
    m_World( m_Gravity )
{
    //ctor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
World::~World()
{
    //dtor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
b2Body* World::CreateBody( b2BodyDef* def )
{
    return m_World.CreateBody( def );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool World::PhysisStep()
{
    m_World.Step(1/60.f, 8, 3);
    return true;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool World::Draw( sf::RenderWindow& window )
{
    for( b2Body* BodyIterator = m_World.GetBodyList(); BodyIterator != 0; BodyIterator = BodyIterator->GetNext() )
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
}
