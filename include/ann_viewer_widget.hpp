#ifndef ANN_VIEWER_WIDGET_HPP
#define ANN_VIEWER_WIDGET_HPP

#include <imgui.h>
#include <imgui-SFML.h>
#include <sigslot.h>
#include <smartpointer.h>
#include <serializable_object.hpp>
#include <entity.hpp>

class AnnViewerWidget : public sigslot::has_slots<>
{
    public:
                AnnViewerWidget();
       virtual ~AnnViewerWidget() = default;

        void SetObject( smart_ptr<codeframe::ObjectNode> obj );
        void Draw( const char* title, bool* p_open = nullptr );

    private:
        static constexpr unsigned int SCREEN_WIDTH = 420U;
        static constexpr unsigned int SCREEN_HEIGHT = 420U;

        ImVec2             m_cursorPos;
        sf::RectangleShape m_rectangle;
        sf::RenderTexture  m_displayTexture;
        sf::RenderStates   m_renderStates;
        smart_ptr<Entity>  m_objEntity;
};


#endif // ANN_VIEWER_WIDGET_HPP
