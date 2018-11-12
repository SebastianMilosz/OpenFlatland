#ifndef COLORIZECIRCLESHAPE_HPP_INCLUDED
#define COLORIZECIRCLESHAPE_HPP_INCLUDED

#include <SFML/Graphics/Export.hpp>
#include <SFML/Graphics/Drawable.hpp>
#include <SFML/Graphics/Transformable.hpp>
#include <SFML/Graphics/VertexArray.hpp>
#include <SFML/System/Vector2.hpp>

namespace sf
{

/*****************************************************************************/
/**
  * @brief Base class for textured shapes with outline
 **
******************************************************************************/
class ColorizeCircleShape : public Drawable, public Transformable
{
public:
             ColorizeCircleShape(float radius = 0, std::size_t pointCount = 30);
    virtual ~ColorizeCircleShape();

    void setTexture(const Texture* texture, bool resetRect = false);
    void setTextureRect(const IntRect& rect);
    void setFillColor(const Color& color);
    void setOutlineColor(const Color& color);
    void setOutlineColor(const std::vector<float>& floatVevtor);
    void setOutlineThickness(float thickness);
    const Texture* getTexture() const;
    const IntRect& getTextureRect() const;
    const Color& getFillColor() const;
    float getOutlineThickness() const;
    FloatRect getLocalBounds() const;
    FloatRect getGlobalBounds() const;

    void setRadius(float radius);
    float getRadius() const;
    void setPointCount(std::size_t count);

    Color* getOutlineColors();
    std::size_t getOutlineColorsCount() const;

    virtual std::size_t getPointCount() const;
    virtual Vector2f getPoint(std::size_t index) const;

protected:
    void update();

private:
    virtual void draw(RenderTarget& target, RenderStates states) const;
    void updateFillColors();
    void updateTexCoords();
    void updateOutlineColors();
    void updateOutline();

private:
    const Texture* m_texture;          ///< Texture of the shape
    IntRect        m_textureRect;      ///< Rectangle defining the area of the source texture to display
    Color          m_fillColor;        ///< Fill color
    float          m_outlineThickness; ///< Thickness of the shape's outline
    VertexArray    m_vertices;         ///< Vertex array containing the fill geometry
    VertexArray    m_outlineVertices;  ///< Vertex array containing the outline geometry
    FloatRect      m_insideBounds;     ///< Bounding rectangle of the inside (fill)
    FloatRect      m_bounds;           ///< Bounding rectangle of the whole shape (outline + fill)
    float          m_radius;           ///< Radius of the circle
    std::size_t    m_pointCount;       ///< Number of points composing the circle
    Color*         m_colorData;        ///<
};

} // namespace sf

#endif // COLORIZECIRCLESHAPE_HPP_INCLUDED
