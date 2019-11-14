#ifndef ANNVIEWERWIDGET_HPP_INCLUDED
#define ANNVIEWERWIDGET_HPP_INCLUDED

#include <imgui.h>
#include <imgui-SFML.h>
#include <sigslot.h>
#include <smartpointer.h>
#include <serializable_object.hpp>

class AnnViewerWidget : public sigslot::has_slots<>
{
    public:
        AnnViewerWidget();
       ~AnnViewerWidget();

        void SetObject( smart_ptr<codeframe::ObjectNode> obj );
        void Draw(const char* title, bool* p_open = NULL);

    private:
        smart_ptr<codeframe::ObjectNode> m_obj;
};


#endif // ANNVIEWERWIDGET_HPP_INCLUDED
