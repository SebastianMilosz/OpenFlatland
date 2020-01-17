#ifndef INFORMATIONWIDGET_HPP_INCLUDED
#define INFORMATIONWIDGET_HPP_INCLUDED

#include <imgui.h>
#include <imgui-SFML.h>
#include <sigslot.h>
#include <smartpointer.h>
#include <serializable_object.hpp>

#include "entity_factory.hpp"

class InformationWidget : public sigslot::has_slots<>
{
    public:
                InformationWidget( sf::RenderWindow& window );
       virtual ~InformationWidget() = default;

        int GetFps();

        std::string FpsToString();

        void Clear();
        void Draw(const char* title, bool* p_open = NULL);

        void SetEntityFactory( const EntityFactory& factory );

    private:
        sf::RenderWindow& m_window;
        const EntityFactory* m_EntityFactory;

        int m_curFps;
        int m_minFps;
        int m_maxFps;
};

#endif // INFORMATIONWIDGET_HPP_INCLUDED
