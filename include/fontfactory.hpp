#ifndef FONTFACTORY_HPP
#define FONTFACTORY_HPP

#include <SFML/Graphics.hpp>
#include <serializable.hpp>

class FontFactory : public codeframe::cSerializable
{
        CODEFRAME_META_CLASS_NAME( "FontFactory" );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
        FontFactory( std::string name, cSerializableInterface* parent );
        virtual ~FontFactory();

        static sf::Font& GetFont();

    protected:

    private:
        static bool m_initialized;
        static sf::Font m_font;
};

#endif // FONTFACTORY_HPP
