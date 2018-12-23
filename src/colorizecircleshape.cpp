#include "colorizecircleshape.hpp"

#include <SFML/Graphics/Shape.hpp>
#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/System/Err.hpp>
#include <cmath>

#include "colorizerealnbr.hpp"

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

    /*****************************************************************************/
    /**
      * @brief Compute the dot product of two vectors
     **
    ******************************************************************************/
    float dotProduct(const sf::Vector2f& p1, const sf::Vector2f& p2)
    {
        return p1.x * p2.x + p1.y * p2.y;
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
    void ColorizeCircleShape::setTexture(const Texture* texture, bool resetRect)
    {
        if (texture)
        {
            // Recompute the texture area if requested, or if there was no texture & rect before
            if (resetRect || (!m_texture && (m_textureRect == IntRect())))
                setTextureRect(IntRect(0, 0, texture->getSize().x, texture->getSize().y));
        }

        // Assign the new texture
        m_texture = texture;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    const Texture* ColorizeCircleShape::getTexture() const
    {
        return m_texture;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void ColorizeCircleShape::setTextureRect(const IntRect& rect)
    {
        m_textureRect = rect;
        updateTexCoords();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    const IntRect& ColorizeCircleShape::getTextureRect() const
    {
        return m_textureRect;
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
        for (std::size_t i = 0; i < m_pointCount; ++i)
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
    void ColorizeCircleShape::setOutlineColor( const std::vector<float>& floatVevtor )
    {
        if ( getPointCount() != floatVevtor.size() )
        {
            setPointCount( floatVevtor.size() );
        }

        ColorizeRealNumbers cl;
        cl.Colorize_Grayscale( floatVevtor, m_colorData, getPointCount() );
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
    ColorizeCircleShape::ColorizeCircleShape(float radius, std::size_t pointCount) :
    m_texture         (NULL),
    m_textureRect     (),
    m_fillColor       (255, 255, 255),
    m_outlineThickness(0),
    m_vertices        (TriangleFan),
    m_outlineVertices (TriangleStrip),
    m_insideBounds    (),
    m_bounds          (),
    m_radius          (radius),
    m_pointCount      (pointCount),
    m_colorData       (NULL)
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
        std::size_t count = getPointCount();
        if (count < 3)
        {
            m_vertices.resize(0);
            m_outlineVertices.resize(0);
            return;
        }

        m_vertices.resize(count + 2); // + 2 for center and repeated first point

        // Position
        for (std::size_t i = 0; i < count; ++i)
            m_vertices[i + 1].position = getPoint(i);
        m_vertices[count + 1].position = m_vertices[1].position;

        // Update the bounding rectangle
        m_vertices[0] = m_vertices[1]; // so that the result of getBounds() is correct
        m_insideBounds = m_vertices.getBounds();

        // Compute the center and make it the first vertex
        m_vertices[0].position.x = m_insideBounds.left + m_insideBounds.width / 2;
        m_vertices[0].position.y = m_insideBounds.top + m_insideBounds.height / 2;

        // Color
        updateFillColors();

        // Texture coordinates
        updateTexCoords();

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

        // Render the inside
        states.texture = m_texture;
        target.draw(m_vertices, states);

        // Render the outline
        if (m_outlineThickness != 0)
        {
            states.texture = NULL;
            target.draw(m_outlineVertices, states);
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void ColorizeCircleShape::updateFillColors()
    {
        for (std::size_t i = 0; i < m_vertices.getVertexCount(); ++i)
            m_vertices[i].color = m_fillColor;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void ColorizeCircleShape::updateTexCoords()
    {
        for (std::size_t i = 0; i < m_vertices.getVertexCount(); ++i)
        {
            float xratio = m_insideBounds.width > 0 ? (m_vertices[i].position.x - m_insideBounds.left) / m_insideBounds.width : 0;
            float yratio = m_insideBounds.height > 0 ? (m_vertices[i].position.y - m_insideBounds.top) / m_insideBounds.height : 0;
            m_vertices[i].texCoords.x = m_textureRect.left + m_textureRect.width * xratio;
            m_vertices[i].texCoords.y = m_textureRect.top + m_textureRect.height * yratio;
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void ColorizeCircleShape::updateOutline()
    {
        std::size_t count = m_vertices.getVertexCount() - 2;
        m_outlineVertices.resize((count + 1) * 2);

        // Recreate color table
        if ( NULL != m_colorData )
        {
            delete[] m_colorData;
            m_colorData = NULL;
        }

        m_colorData = new Color[ m_pointCount ];

        for (std::size_t i = 0; i < count; ++i)
        {
            std::size_t index = i + 1;

            // Get the two segments shared by the current point
            Vector2f p0 = (i == 0) ? m_vertices[count].position : m_vertices[index - 1].position;
            Vector2f p1 = m_vertices[index].position;
            Vector2f p2 = m_vertices[index + 1].position;

            // Compute their normal
            Vector2f n1 = computeNormal(p0, p1);
            Vector2f n2 = computeNormal(p1, p2);

            // Make sure that the normals point towards the outside of the shape
            // (this depends on the order in which the points were defined)
            if (dotProduct(n1, m_vertices[0].position - p1) > 0)
                n1 = -n1;
            if (dotProduct(n2, m_vertices[0].position - p1) > 0)
                n2 = -n2;

            // Combine them to get the extrusion direction
            float factor = 1.f + (n1.x * n2.x + n1.y * n2.y);
            Vector2f normal = (n1 + n2) / factor;

            // Update the outline points
            m_outlineVertices[i * 2 + 0].position = p1;
            m_outlineVertices[i * 2 + 1].position = p1 + normal * m_outlineThickness;
        }

        // Duplicate the first point at the end, to close the outline
        m_outlineVertices[count * 2 + 0].position = m_outlineVertices[0].position;
        m_outlineVertices[count * 2 + 1].position = m_outlineVertices[1].position;

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
        std::size_t count = m_vertices.getVertexCount() - 2;
        unsigned int n = 0;
        Color cl;
        for (std::size_t i = 0; i < count; ++i)
        {
            cl = m_colorData[i];
            m_outlineVertices[n + 0].color = cl;
            m_outlineVertices[n + 1].color = cl;
            n += 2;
        }

        m_outlineVertices[count * 2 + 0].color = cl;
        m_outlineVertices[count * 2 + 1].color = cl;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void ColorizeCircleShape::setRadius(float radius)
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
    void ColorizeCircleShape::setPointCount(std::size_t count)
    {
        m_pointCount = count;
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
    Vector2f ColorizeCircleShape::getPoint(std::size_t index) const
    {
        static const float pi = 3.141592654F;

        float angle = -((index * 2.0F * pi / m_pointCount) + (pi / 2.0F));
        float x = std::cos( angle ) * m_radius;
        float y = std::sin( angle ) * m_radius;

        return Vector2f(m_radius + x, m_radius + y);
    }

} // namespace sf
