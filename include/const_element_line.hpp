#ifndef CONSTELEMENTLINE_HPP_INCLUDED
#define CONSTELEMENTLINE_HPP_INCLUDED

#include <serializable_object.hpp>

#include "const_element.hpp"
#include "drawable_object.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
class ConstElementLine : public ConstElement, public DrawableObject
{
        CODEFRAME_META_CLASS_NAME( "ConstElementLine" );
        CODEFRAME_META_BUILD_TYPE( codeframe::DYNAMIC );
        CODEFRAME_META_CONSTRUCT_PATERN( "SPoint,EPoint" );

    public:
        ConstElementLine( const std::string& name, codeframe::Point2D<int>& startPoint, codeframe::Point2D<int>& endPoint );
        virtual ~ConstElementLine();
        ConstElementLine(const ConstElementLine& other);
        ConstElementLine& operator=(const ConstElementLine& other);

        void draw( sf::RenderTarget& target, sf::RenderStates states ) const override;
        void Draw( sf::RenderWindow& window );
        void Synchronize( b2Body* body ) override;

        codeframe::Property< codeframe::Point2D<int> > StartPoint;
        codeframe::Property< codeframe::Point2D<int> > EndPoint;

    private:
        sf::Vertex m_line[2];
};

#endif // CONSTELEMENTLINE_HPP_INCLUDED
