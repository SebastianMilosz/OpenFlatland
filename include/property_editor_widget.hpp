#ifndef PROPERTY_EDITOR_WIDGET_HPP
#define PROPERTY_EDITOR_WIDGET_HPP

#include <imgui.h>
#include <imgui-SFML.h>
#include <sigslot.h>
#include <smartpointer.h>
#include <serializable_object.hpp>
#include <serializable_object_container.hpp>

class PropertyEditorWidget : public sigslot::has_slots<>
{
    public:
        PropertyEditorWidget();
       ~PropertyEditorWidget() = default;

        void Clear();
        void SetObject(smart_ptr<codeframe::ObjectNode> obj);
        void Draw(const char* title, bool* p_open = NULL);

    private:
        void ShowHelpMarker(const char* desc );
        void ShowObject(smart_ptr<codeframe::ObjectNode> obj);
        void ShowRawObject(smart_ptr<codeframe::ObjectNode> obj);
        void ShowRawProperty(codeframe::PropertyBase* prop);

        codeframe::VariantValue ShowSelectParameterDialog(smart_ptr<codeframe::ObjectNode> obj,
                                              codeframe::eType typeFilter = codeframe::eType::TYPE_NON);

        smart_ptr<codeframe::ObjectNode> m_obj;
        bool_t valueChangeGraphEnable = false;
};

#endif // PROPERTY_EDITOR_WIDGET_HPP
