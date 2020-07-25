#ifndef COLORIZECIRCLESHAPE_HPP_INCLUDED
#define COLORIZECIRCLESHAPE_HPP_INCLUDED

#include <SFML/Graphics/Export.hpp>
#include <SFML/Graphics/Drawable.hpp>
#include <SFML/Graphics/Transformable.hpp>
#include <SFML/Graphics/VertexArray.hpp>
#include <SFML/System/Vector2.hpp>

#include "entity_vision_node.hpp"

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
                 ColorizeCircleShape(float radius = 0, std::size_t pointCount = 30, const int startAngle = 0, const int endAngle = 360);
        virtual ~ColorizeCircleShape();

        void setFillColor(const Color& color);
        void setOutlineColor(const Color& color);
        void setOutlineColor(const thrust::host_vector<RayData>& floatVevtor);
        void setOutlineThickness(float thickness);
        const Color& getFillColor() const;
        float getOutlineThickness() const;
        FloatRect getLocalBounds() const;
        FloatRect getGlobalBounds() const;

        void setRadius(const float radius);
        float getRadius() const;
        void setPointCount(const std::size_t count);

        void setStartAngle(const int startAngle);
        void setEndAngle(const int endAngle);

        void setColorizeMode(const int mode);

        Color* getOutlineColors();
        std::size_t getOutlineColorsCount() const;

        virtual std::size_t getPointCount() const;
        virtual Vector2f getPoint(const std::size_t index) const;

    protected:
        void update();

    private:
        virtual void draw(RenderTarget& target, RenderStates states ) const;
        void updateFillColors();
        void updateOutlineColors();
        void updateOutline();

    private:
        Color          m_fillColor;        ///< Fill color
        float          m_outlineThickness; ///< Thickness of the shape's outline
        VertexArray    m_outlineVertices;  ///< Vertex array containing the outline geometry
        FloatRect      m_insideBounds;     ///< Bounding rectangle of the inside (fill)
        FloatRect      m_bounds;           ///< Bounding rectangle of the whole shape (outline + fill)
        float          m_radius;           ///< Radius of the circle
        std::size_t    m_pointCount;       ///< Number of points composing the circle
        Color*         m_colorData;        ///<
        int            m_colorizeMode;

        int            m_StartAngle;
        int            m_EndAngle;
};

} // namespace sf

#endif // COLORIZECIRCLESHAPE_HPP_INCLUDED
