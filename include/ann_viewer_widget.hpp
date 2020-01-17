#ifndef ANN_VIEWER_WIDGET_HPP
#define ANN_VIEWER_WIDGET_HPP

#include <imgui.h>
#include <imgui-SFML.h>
#include <sigslot.h>
#include <smartpointer.h>
#include <serializable_object.hpp>

class AnnViewerWidget : public sigslot::has_slots<>
{
    public:
                AnnViewerWidget();
       virtual ~AnnViewerWidget() = default;

        void SetObject( smart_ptr<codeframe::ObjectNode> obj );
        void Draw( const char* title, bool* p_open = nullptr );

    private:
        smart_ptr<codeframe::ObjectNode> m_obj;
};


#endif // ANN_VIEWER_WIDGET_HPP
