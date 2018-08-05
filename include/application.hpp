#ifndef APPLICATION_HPP_INCLUDED
#define APPLICATION_HPP_INCLUDED

#include <serializable.h>

#include "world.h"
#include "entityfactory.h"
#include "constelementsfactory.hpp"
#include "guiwidgetslayer.h"
#include "entity.h"

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

            sf::RenderWindow&   m_Window;
            GUIWidgetsLayer     m_Widgets;
            World               m_World;
            EntityFactory       m_Factory;

};

#endif // APPLICATION_HPP_INCLUDED
