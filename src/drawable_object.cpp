#include "drawable_object.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
DrawableObject::DrawableObject()
{
    //ctor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void DrawableObject::draw( sf::RenderTarget& target, sf::RenderStates states ) const
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void DrawableObject::appendQuad(sf::VertexArray& vec, const sf::Vector2f position, const sf::Vector2f size, const sf::Color color)
{
    // Define the 4 corners based on position and size
    sf::Vector2f topLeft     = position;
    sf::Vector2f topRight    = {position.x + size.x, position.y};
    sf::Vector2f bottomRight = {position.x + size.x, position.y + size.y};
    sf::Vector2f bottomLeft  = {position.x, position.y + size.y};

    // Helper vertex to reuse
    sf::Vertex v;
    v.color = color;

    // --- Triangle 1 ---
    v.position = topLeft;
    vec.append(v);

    v.position = topRight;
    vec.append(v);

    v.position = bottomRight;
    vec.append(v);

    // --- Triangle 2 ---
    v.position = bottomRight;
    vec.append(v);

    v.position = bottomLeft;
    vec.append(v);

    v.position = topLeft;
    vec.append(v);
}
