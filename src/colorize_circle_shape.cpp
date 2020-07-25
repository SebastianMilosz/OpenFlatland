#include "colorize_circle_shape.hpp"

#include <SFML/Graphics/Shape.hpp>
#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/System/Err.hpp>
#include <cmath>

#include "colorize_ray_data.hpp"

namespace
{
    /*****************************************************************************/
    /**
      * @brief Compute the normal of a segment
     **
    ******************************************************************************/
    sf::Vector2f computeNormal(const sf::Vector2f& p1, const sf::Vector2f& p2)
    {
        sf::Vector2f normal(p1.y - p2.y, p2.x - p1.x);
        float length = std::sqrt(normal.x * normal.x + normal.y * normal.y);
        if (length != 0.f)
            normal /= length;
        return normal;
    }
}

namespace sf
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    ColorizeCircleShape::~ColorizeCircleShape()
    {
        if ( NULL != m_colorData )
        {
            delete[] m_colorData;
            m_colorData = NULL;
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void ColorizeCircleShape::setFillColor(const Color& color)
    {
        m_fillColor = color;
        updateFillColors();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    const Color& ColorizeCircleShape::getFillColor() const
    {
        return m_fillColor;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void ColorizeCircleShape::setOutlineColor(const Color& color)
    {
        for ( std::size_t i = 0; i < m_pointCount; ++i )
        {
            m_colorData[i] = color;
        }
        updateOutlineColors();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void ColorizeCircleShape::setOutlineColor(const thrust::host_vector<RayData>& floatVevtor)
    {
        if ( m_pointCount != floatVevtor.size() )
        {
            setPointCount( floatVevtor.size() );
        }

        ColorizeRayData cl;
        cl.Colorize( (ColorizeRayData::eColorizeMode)m_colorizeMode, floatVevtor, m_colorData, m_pointCount );
        updateOutlineColors();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void ColorizeCircleShape::setOutlineThickness(float thickness)
    {
        m_outlineThickness = thickness;
        update(); // recompute everything because the whole shape must be offset
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    float ColorizeCircleShape::getOutlineThickness() const
    {
        return m_outlineThickness;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    FloatRect ColorizeCircleShape::getLocalBounds() const
    {
        return m_bounds;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    FloatRect ColorizeCircleShape::getGlobalBounds() const
    {
        return getTransform().transformRect(getLocalBounds());
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    ColorizeCircleShape::ColorizeCircleShape(float radius, std::size_t pointCount, const int startAngle, const int endAngle) :
    m_fillColor       (255, 255, 255),
    m_outlineThickness(0),
    m_outlineVertices (TriangleStrip),
    m_insideBounds    (),
    m_bounds          (),
    m_radius          (radius),
    m_pointCount      (pointCount),
    m_colorData       (NULL),
    m_colorizeMode    (ColorizeRayData::eColorizeMode::Grayscale),
    m_StartAngle      (startAngle),
    m_EndAngle        (endAngle)
    {

    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void ColorizeCircleShape::update()
    {
        // Get the total number of points of the shape
        if ( m_pointCount < 3 )
        {
            m_outlineVertices.resize(0);
            return;
        }

        // Outline
        updateOutline();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void ColorizeCircleShape::draw(RenderTarget& target, RenderStates states) const
    {
        states.transform *= getTransform();

        // Render the outline
        if (m_outlineThickness != 0)
        {
            states.texture = nullptr;
            target.draw( m_outlineVertices, states );
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void ColorizeCircleShape::updateFillColors()
    {
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void ColorizeCircleShape::updateOutline()
    {
        m_outlineVertices.resize( m_pointCount * 2U );

        // Recreate color table
        if ( NULL != m_colorData )
        {
            delete[] m_colorData;
            m_colorData = NULL;
        }

        m_colorData = new Color[ m_pointCount ];

        for ( std::size_t i = 0U; i < m_pointCount; ++i )
        {
            std::size_t index( i + 1U );

            // Get the two segments shared by the current point
            Vector2f p0( ( i == 0U ) ? getPoint( m_pointCount ) : getPoint( index - 1U ) );
            Vector2f p1( getPoint( index ) );
            Vector2f p2( getPoint( index + 1U ) );

            // Compute their normal
            Vector2f n1( computeNormal(p0, p1) );
            Vector2f n2( computeNormal(p1, p2) );

            Vector2f normal( n1 + n2 ); //factor

            // Update the outline points
            m_outlineVertices[i * 2U + 0U].position = p1;
            m_outlineVertices[i * 2U + 1U].position = p1 + normal * m_outlineThickness;
        }

        // Update outline colors
        updateOutlineColors();

        // Update the shape's bounds
        m_bounds = m_outlineVertices.getBounds();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void ColorizeCircleShape::updateOutlineColors()
    {
        uint32_t n( 0U );
        Color cl;
        for ( std::size_t i = 0U; i < m_pointCount; ++i )
        {
            cl = m_colorData[i];
            m_outlineVertices[n + 0U].color = cl;
            m_outlineVertices[n + 1U].color = cl;
            n += 2U;
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void ColorizeCircleShape::setRadius(const float radius)
    {
        m_radius = radius;
        update();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    float ColorizeCircleShape::getRadius() const
    {
        return m_radius;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void ColorizeCircleShape::setPointCount(const std::size_t count)
    {
        m_pointCount = count;
        update();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void ColorizeCircleShape::setStartAngle(const int startAngle)
    {
        m_StartAngle = startAngle;
        update();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void ColorizeCircleShape::setEndAngle(const int endAngle)
    {
        m_EndAngle = endAngle;
        update();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void ColorizeCircleShape::setColorizeMode(const int mode)
    {
        m_colorizeMode = mode;
        update();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    Color* ColorizeCircleShape::getOutlineColors()
    {
        return m_colorData;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::size_t ColorizeCircleShape::getOutlineColorsCount() const
    {
        return m_pointCount;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::size_t ColorizeCircleShape::getPointCount() const
    {
        return m_pointCount;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    Vector2f ColorizeCircleShape::getPoint(const std::size_t index) const
    {
        static const float pi = 3.141592654F;

        // angle( -((index * 360o / m_pointCount) + 90o) );
        float angleRange( std::max(m_StartAngle,m_EndAngle) - std::min(m_StartAngle,m_EndAngle) );
        float angle( ((index * ( angleRange* (pi/180.0F)) / m_pointCount) + ((std::min(m_StartAngle,m_EndAngle)+90.0F) * (pi/180.0F))) );
        float x( std::cos( angle ) * m_radius );
        float y( std::sin( angle ) * m_radius );

        return Vector2f(m_radius + x, m_radius + y);
    }

} // namespace sf
