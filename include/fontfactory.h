#ifndef FONTFACTORY_H
#define FONTFACTORY_H

#include <serializable.h>
#include <SFML/Graphics.hpp>

class FontFactory : public codeframe::cSerializable
{
    public:
        std::string Role()            const { return "Object";      }
        std::string Class()           const { return "FontFactory"; }
        std::string BuildType()       const { return "Static";      }
        std::string ConstructPatern() const { return "";            }

    public:
        FontFactory( std::string name, cSerializableInterface* parent );
        virtual ~FontFactory();

        static sf::Font& GetFont();

    protected:

    private:
        static bool m_initialized;
        static sf::Font m_font;
};

#endif // FONTFACTORY_H
