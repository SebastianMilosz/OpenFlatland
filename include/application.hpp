#ifndef APPLICATION_HPP_INCLUDED
#define APPLICATION_HPP_INCLUDED

#include <serializable.h>

class Application : public codeframe::cSerializable
{
    public:
        std::string Role()            const { return "Object";      }
        std::string Class()           const { return "Application"; }
        std::string BuildType()       const { return "Static";      }
        std::string ConstructPatern() const { return ""; }

    public:
                 Application( std::string name );
        virtual ~Application();

};

#endif // APPLICATION_HPP_INCLUDED
