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
        static constexpr unsigned int MENU_LEFT_OFFSET = 200U;
        static constexpr unsigned int SCREEN_WIDTH = 520U;
        static constexpr unsigned int SCREEN_HEIGHT = 520U;

        sf::RenderTexture  m_displayTexture;
        sf::RenderStates   m_renderStates;
        ImVec2             m_cursorPos;
        smart_ptr<Entity>  m_objEntity;
};


#endif // ANN_VIEWER_WIDGET_HPP
