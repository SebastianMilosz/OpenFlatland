#ifndef GUIWIDGETSLAYER_H
#define GUIWIDGETSLAYER_H

#include <SFML/Graphics.hpp>

class GUIWidgetsLayer
{
    public:
        enum { MOUSE_MODE_SEL_ENTITY, MOUSE_MODE_DEL_ENTITY, MOUSE_MODE_ADD_ENTITY };

    public:
        GUIWidgetsLayer( sf::RenderWindow& window );
        virtual ~GUIWidgetsLayer();

        void AddGuiRegion( int x, int y, int w, int h );

        bool MouseOnGui();
        bool HandleEvent( sf::Event& event );
        void Draw();

        void        SetMouseModeString( std::string mode );
        std::string GetMouseModeString();

        void        SetMouseModeId( int mode );
        int         GetMouseModeId();

        int GetFps();

    protected:

    private:
        sf::RenderWindow&   m_window;
        sf::Clock           m_deltaClock;

        int                 m_MouseMode;
        bool                m_mouseCapturedByGui;

        std::vector< sf::Rect<int> > m_guiRegions;
};

#endif // GUIWIDGETSLAYER_H
