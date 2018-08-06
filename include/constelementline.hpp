#ifndef CONSTELEMENTLINE_HPP_INCLUDED
#define CONSTELEMENTLINE_HPP_INCLUDED

#include <serializable.h>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
class ConstElementLine : public ConstElement
{
    public:
        std::string Role()      const { return "Object";            }
        std::string Class()     const { return "ConstElementLine";  }
        std::string BuildType() const { return "Dynamic";           }

    public:
        ConstElementLine( std::string name, int x, int y, int z );
        virtual ~ConstElementLine();
        ConstElementLine(const ConstElementLine& other);
        ConstElementLine& operator=(const ConstElementLine& other);
};

#endif // CONSTELEMENTLINE_HPP_INCLUDED
