#ifndef INFORMATIONWIDGET_HPP_INCLUDED
#define INFORMATIONWIDGET_HPP_INCLUDED

#include <imgui.h>
#include <imgui-SFML.h>
#include <sigslot.h>
#include <smartpointer.h>
#include <serializable.hpp>

#include "entityfactory.hpp"

class InformationWidget : public sigslot::has_slots<>
{
    public:
        InformationWidget( sf::RenderWindow& window );
       ~InformationWidget();

        int GetFps();

        void Clear();
        void Draw(const char* title, bool* p_open = NULL);

        void SetEntityFactory( const EntityFactory& factory );

    private:
        sf::RenderWindow& m_window;
        const EntityFactory* m_EntityFactory;
};

#endif // INFORMATIONWIDGET_HPP_INCLUDED
