#ifndef APPLICATION_HPP_INCLUDED
#define APPLICATION_HPP_INCLUDED

#include <serializable.hpp>

#include "world.hpp"
#include "entityfactory.hpp"
#include "constelementsfactory.hpp"
#include "fontfactory.hpp"
#include "guiwidgetslayer.hpp"
#include "entity.hpp"

class Application : public codeframe::cSerializable
{
    public:
        std::string Role()            const { return "Object";      }
        std::string Class()           const { return "Application"; }
        std::string BuildType()       const { return "Static";      }
        std::string ConstructPatern() const { return ""; }

    public:
                 Application( std::string name, sf::RenderWindow& window );
        virtual ~Application();

        void ProcesseEvents( sf::Event& event );
        void ProcesseLogic( void );

    private:
            void ZoomViewAt( sf::Vector2i pixel, sf::RenderWindow& window, float zoom );

            const float         m_zoomAmount;

            std::string         m_cfgFilePath;
            std::string         m_perFilePath;

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
