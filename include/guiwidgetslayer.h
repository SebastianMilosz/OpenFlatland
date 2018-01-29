#ifndef GUIWIDGETSLAYER_H
#define GUIWIDGETSLAYER_H

#include <SFML/Graphics.hpp>
#include <TGUI/TGUI.hpp>

class GUIWidgetsLayer
{
    public:
        GUIWidgetsLayer( sf::RenderWindow& window );
        virtual ~GUIWidgetsLayer();

        void AddGuiRegion( int x, int y, int w, int h );

        bool MouseOnGui();
        bool HandleEvent( sf::Event& event );
        void Draw();

        void SetMode( int mode );
        int  GetMode();

    protected:

    private:
        tgui::Gui           m_gui;
        sf::RenderWindow&   m_window;

        int m_mode;

        std::vector< sf::Rect<int> > m_guiRegions;
};

#endif // GUIWIDGETSLAYER_H
