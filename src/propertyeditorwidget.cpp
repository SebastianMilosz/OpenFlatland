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
        uint32_t uid = obj->Identity().GetUId();

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
            for ( codeframe::PropertyIterator it = obj->PropertyManager().begin(); it != obj->PropertyManager().end(); ++it )
            {
                codeframe::PropertyBase* iser = *it;

                if ( (codeframe::PropertyBase*)NULL != iser )
                {
                    uint32_t upropid = iser->Id();

                    // Use property pointer as identifier.
                    ImGui::PushID( upropid );

                    ImGui::AlignTextToFramePadding();

                    ImGui::Bullet();
                    ImGui::Selectable( iser->Name().c_str() );

                    ImGui::NextColumn();
                    ImGui::PushItemWidth(-1);

                    // Depend on the property type
                    switch ( iser->Info().GetKind() )
                    {
                        case codeframe::KIND_NON:
                        {

                            break;
                        }
                        case codeframe::KIND_LOGIC:
                        {
                            bool check = (bool)(*iser);
                            ImGui::Checkbox("##value", &check);
                            (*iser) = check;
                            break;
                        }
                        case codeframe::KIND_NUMBER:
                        {
                            int value = (int)(*iser);
                            ImGui::InputInt("##value", &value, 1);
                            (*iser) = value;
                            break;
                        }
                        case codeframe::KIND_NUMBERRANGE:
                        {

                            break;
                        }
                        case codeframe::KIND_REAL:
                        {
                            float value = (float)(*iser);
                            ImGui::InputFloat("##value", &value, 1.0f);
                            (*iser) = value;
                            break;
                        }
                        case codeframe::KIND_TEXT:
                        {
                            char newText[32];
                            memset(newText, 0, 32);
                            std::string textValue = (std::string)(*iser);
                            strncpy(newText, textValue.c_str(), textValue.length());
                            ImGui::InputText("##value", newText, IM_ARRAYSIZE(newText));
                            (*iser) = std::string( newText );
                            break;
                        }
                        case codeframe::KIND_ENUM:
                        {

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
                    }

                    ImGui::PopItemWidth();
                    ImGui::NextColumn();
                    ImGui::PopID();
                }
            }
            ImGui::TreePop();
        }
        ImGui::PopID();
    }
}
