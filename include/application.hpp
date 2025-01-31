#ifndef APPLICATION_HPP_INCLUDED
#define APPLICATION_HPP_INCLUDED

#include <serializable_object.hpp>

#include "world.hpp"
#include "entity_factory.hpp"
#include "const_elements_factory.hpp"
#include "font_factory.hpp"
#include "gui_widgets_layer.hpp"
#include "entity.hpp"

class Application : public codeframe::Object
{
        CODEFRAME_META_CLASS_NAME("Application");
        CODEFRAME_META_BUILD_TYPE(codeframe::STATIC);

    public:
                 Application(std::string name, sf::RenderWindow& window);
        virtual ~Application() = default;

        void ProcesseEvents(const std::optional<sf::Event>& event);
        void ProcesseLogic();

    private:
            void ZoomViewAt(sf::Vector2i pixel, sf::RenderWindow& window, const float zoom);

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
