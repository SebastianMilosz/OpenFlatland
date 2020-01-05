#ifndef PROPERTYEDITORWIDGET_HPP_INCLUDED
#define PROPERTYEDITORWIDGET_HPP_INCLUDED

#include <imgui.h>
#include <imgui-SFML.h>
#include <sigslot.h>
#include <smartpointer.h>
#include <serializable_object.hpp>

class PropertyEditorWidget : public sigslot::has_slots<>
{
    public:
        PropertyEditorWidget();
       ~PropertyEditorWidget();

        void Clear();
        void SetObject( smart_ptr<codeframe::ObjectNode> obj );
        void Draw(const char* title, bool* p_open = NULL);

    private:
        void ShowHelpMarker( const char* desc );
        void ShowObject( smart_ptr<codeframe::ObjectNode> obj );
        void ShowRawObject( codeframe::ObjectNode* obj );
        void ShowRawProperty( codeframe::PropertyBase* prop );

        smart_ptr<codeframe::ObjectNode> m_obj;
};

#endif // PROPERTYEDITORWIDGET_HPP_INCLUDED