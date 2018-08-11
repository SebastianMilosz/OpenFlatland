#ifndef CONSTELEMENTLINE_HPP_INCLUDED
#define CONSTELEMENTLINE_HPP_INCLUDED

#include "constelement.hpp"

#include <serializable.h>
#include <extendedtypepoint2d.hpp>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
class ConstElementLine : public ConstElement
{
    public:
        std::string Role()            const { return "Object";           }
        std::string Class()           const { return "ConstElementLine"; }
        std::string BuildType()       const { return "Dynamic";          }
        std::string ConstructPatern() const { return "SPoint,EPoint";    }

    public:
        ConstElementLine( std::string name, codeframe::Point2D& startPoint, codeframe::Point2D& endPoint );
        virtual ~ConstElementLine();
        ConstElementLine(const ConstElementLine& other);
        ConstElementLine& operator=(const ConstElementLine& other);

        codeframe::Property<codeframe::Point2D, ConstElementLine> StartPoint;
        codeframe::Property<codeframe::Point2D, ConstElementLine> EndPoint;
};

#endif // CONSTELEMENTLINE_HPP_INCLUDED
