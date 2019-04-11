#ifndef CONSTELEMENTLINE_HPP_INCLUDED
#define CONSTELEMENTLINE_HPP_INCLUDED

#include "constelement.hpp"

#include <serializable.hpp>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
class ConstElementLine : public ConstElement
{
        CODEFRAME_META_CLASS_NAME( "ConstElementLine" );
        CODEFRAME_META_BUILD_TYPE( codeframe::DYNAMIC );
        CODEFRAME_META_CONSTRUCT_PATERN( "SPoint,EPoint" );

    public:
        ConstElementLine( const std::string& name, codeframe::Point2D<int>& startPoint, codeframe::Point2D<int>& endPoint );
        virtual ~ConstElementLine();
        ConstElementLine(const ConstElementLine& other);
        ConstElementLine& operator=(const ConstElementLine& other);

        virtual void Draw( sf::RenderWindow& window, b2Body* body );

        codeframe::Property<codeframe::Point2D<int>, ConstElementLine> StartPoint;
        codeframe::Property<codeframe::Point2D<int>, ConstElementLine> EndPoint;
};

#endif // CONSTELEMENTLINE_HPP_INCLUDED
