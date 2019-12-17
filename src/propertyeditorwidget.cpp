#include "propertyeditorwidget.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
PropertyEditorWidget::PropertyEditorWidget()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
PropertyEditorWidget::~PropertyEditorWidget()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PropertyEditorWidget::Clear()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PropertyEditorWidget::SetObject( smart_ptr<ObjectNode> obj )
{
    m_obj = obj;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PropertyEditorWidget::ShowHelpMarker( const char* desc )
{
    ImGui::TextDisabled("(?)");
    if ( ImGui::IsItemHovered() )
    {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PropertyEditorWidget::Draw( const char* title, bool* p_open )
{
    ImGui::SetNextWindowSize( ImVec2( 430, 450 ), ImGuiCond_FirstUseEver );
    if ( !ImGui::Begin( title, p_open ) )
    {
        ImGui::End();
        return;
    }

    ShowHelpMarker("Help Information");

    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2(2,2) );
    ImGui::Columns( 2 );
    ImGui::Separator();

    ShowObject( m_obj );

    ImGui::Columns( 1 );
    ImGui::Separator();
    ImGui::PopStyleVar();
    ImGui::End();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PropertyEditorWidget::ShowObject( smart_ptr<codeframe::ObjectNode> obj )
{
    if ( smart_ptr_isValid( obj ) == true )
    {
        ShowRawObject( smart_ptr_getRaw( obj ) );
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PropertyEditorWidget::ShowRawObject( codeframe::ObjectNode* obj )
{
    uint32_t uid( obj->Identity().GetUId() );

    // Take object pointer as unique id
    ImGui::PushID( uid );
    ImGui::AlignTextToFramePadding();
    bool node_open = ImGui::TreeNode( "Object", "%s", obj->Identity().ObjectName().c_str() );

    ImGui::NextColumn();
    ImGui::AlignTextToFramePadding();
    ImGui::Text( "Class: %s", obj->Class().c_str() );
    ImGui::NextColumn();

    if ( node_open == true )
    {
        // Iterate through properties in object
        for ( auto it = obj->PropertyList().begin(); it != obj->PropertyList().end(); ++it )
        {
            codeframe::PropertyBase* iser = *it;

            ShowRawProperty( iser );
        }

        // Iterate through childs in the object
        for ( auto it = obj->ChildList().begin(); it != obj->ChildList().end(); ++it )
        {
            auto childObject = *it;

            ShowRawObject( childObject );
        }

        ImGui::TreePop();
    }
    ImGui::PopID();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PropertyEditorWidget::ShowRawProperty( codeframe::PropertyBase* prop )
{
    if ( (codeframe::PropertyBase*)nullptr != prop )
    {
        uint32_t upropid = prop->Id();

        // Use property pointer as identifier.
        ImGui::PushID( upropid );

        ImGui::AlignTextToFramePadding();

        ImGui::Bullet();
        ImGui::Selectable( prop->Name().c_str() );

        ImGui::NextColumn();
        ImGui::PushItemWidth(-1);

        // Depend on the property type
        switch ( prop->Info().GetKind() )
        {
            case codeframe::KIND_NON:
            {

                break;
            }
            case codeframe::KIND_LOGIC:
            {
                bool check = (bool)(*prop);
                ImGui::Checkbox("##value", &check);
                (*prop) = check;
                break;
            }
            case codeframe::KIND_NUMBER:
            {
                int value = (int)(*prop);
                ImGui::InputInt("##value", &value, 1);
                (*prop) = value;
                break;
            }
            case codeframe::KIND_NUMBERRANGE:
            {
                int value = (int)(*prop);
                ImGui::InputInt("##value", &value, 1);
                (*prop) = value;
                break;
            }
            case codeframe::KIND_REAL:
            {
                float value = (float)(*prop);
                ImGui::InputFloat("##value", &value, 1.0f);
                (*prop) = value;
                break;
            }
            case codeframe::KIND_TEXT:
            {
                char newText[32];
                memset(newText, 0, 32);
                std::string textValue = (std::string)(*prop);
                strncpy(newText, textValue.c_str(), textValue.length());
                ImGui::InputText("##value", newText, IM_ARRAYSIZE(newText));
                (*prop) = std::string( newText );
                break;
            }
            case codeframe::KIND_ENUM:
            {
                static ImGuiComboFlags flags = 0;
                // General BeginCombo() API, you have full control over your selection data and display type.
                // (your selection data could be an index, a pointer to the object, an id for the object, a flag stored in the object itself, etc.)
                const char* items[] = { "AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF", "GGGG", "HHHH", "IIII", "JJJJ", "KKKK", "LLLLLLL", "MMMM", "OOOOOOO" };
                static const char* item_current = items[0];            // Here our selection is a single pointer stored outside the object.
                if (ImGui::BeginCombo("combo 1", item_current, flags)) // The second parameter is the label previewed before opening the combo.
                {
                    for (int n = 0; n < IM_ARRAYSIZE(items); n++)
                    {
                        bool is_selected = (item_current == items[n]);
                        if (ImGui::Selectable(items[n], is_selected))
                            item_current = items[n];
                        if (is_selected)
                            ImGui::SetItemDefaultFocus();   // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
                    }
                    ImGui::EndCombo();
                }
                break;
            }
            case codeframe::KIND_DIR:
            {

                break;
            }
            case codeframe::KIND_URL:
            {

                break;
            }
            case codeframe::KIND_FILE:
            {

                break;
            }
            case codeframe::KIND_DATE:
            {

                break;
            }
            case codeframe::KIND_FONT:
            {

                break;
            }
            case codeframe::KIND_COLOR:
            {

                break;
            }
            case codeframe::KIND_IMAGE:
            {

                break;
            }
            default:
            {

            }
        }

        ImGui::PopItemWidth();
        ImGui::NextColumn();
        ImGui::PopID();
    }
}
