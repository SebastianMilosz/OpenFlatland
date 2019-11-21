#ifndef APPLICATION_HPP_INCLUDED
#define APPLICATION_HPP_INCLUDED

#include <serializable_object.hpp>

#include "world.hpp"
#include "entityfactory.hpp"
#include "constelementsfactory.hpp"
#include "fontfactory.hpp"
#include "guiwidgetslayer.hpp"
#include "entity.hpp"

class Application : public codeframe::Object
{
        CODEFRAME_META_CLASS_NAME( "Application" );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
                 Application( std::string name, sf::RenderWindow& window );
        virtual ~Application();

        void ProcesseEvents( sf::Event& event );
        void ProcesseLogic( void );

    private:
            void ZoomViewAt( sf::Vector2i pixel, sf::RenderWindow& window, const float zoom );

            const float         m_zoomAmount;

            std::string         m_apiDir;
            std::string         m_cfgFilePath;
            std::string         m_perFilePath;
            std::string         m_logFilePath;
            std::string         m_guiFilePath;

            sf::RenderWindow&       m_Window;
            GUIWidgetsLayer         m_Widgets;
            World                   m_World;
            EntityFactory           m_EntityFactory;
            ConstElementsFactory    m_ConstElementsFactory;
            FontFactory             m_FontFactory;

            // Temporary
            int lineCreateState;
            sf::Vector2f startPoint;
            sf::Vector2f endPoint;
};

#endif // APPLICATION_HPP_INCLUDED
