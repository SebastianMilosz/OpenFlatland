#ifndef FONTFACTORY_HPP
#define FONTFACTORY_HPP

#include <SFML/Graphics.hpp>
#include <serializable_object.hpp>

class FontFactory : public codeframe::Object
{
        CODEFRAME_META_CLASS_NAME( "FontFactory" );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
        FontFactory( const std::string& name, ObjectNode* parent );
        virtual ~FontFactory();

        static sf::Font& GetFont();

    protected:

    private:
        static bool m_initialized;
        static sf::Font m_font;
};

#endif // FONTFACTORY_HPP
