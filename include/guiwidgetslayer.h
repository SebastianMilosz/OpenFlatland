#ifndef GUIWIDGETSLAYER_H
#define GUIWIDGETSLAYER_H

#include <SFML/Graphics.hpp>
#include <TGUI/TGUI.hpp>

class GUIWidgetsLayer
{
    public:
        GUIWidgetsLayer( sf::RenderWindow& window );
        virtual ~GUIWidgetsLayer();

        bool HandleEvent( sf::Event& event );
        void Draw();

    protected:

    private:
        tgui::Gui m_gui;
};

#endif // GUIWIDGETSLAYER_H
