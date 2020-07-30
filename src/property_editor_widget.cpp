#include "property_editor_widget.hpp"

#include <ray_data.hpp>
#include <thrust/device_vector.h>
#include <imgui_internal.h> // Currently imgui dosn't have disable/enable control feature
#include <MathUtilities.h>

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
        ShowRawObject( obj );
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PropertyEditorWidget::ShowRawObject( smart_ptr<codeframe::ObjectNode> obj )
{
    uint32_t uid( obj->Identity().GetUId() );

    // Take object pointer as unique id
    ImGui::PushID( uid );
    ImGui::AlignTextToFramePadding();

    std::string objectName = obj->Identity().ObjectName();
    bool node_open = false;

    if (obj->Role() == eBuildRole::CONTAINER)
    {
        node_open = ImGui::TreeNode( "Object", "%s[%d]", objectName.c_str(), obj->Count() );
    }
    else
    {
        node_open = ImGui::TreeNode( "Object", "%s", objectName.c_str() );
    }

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
template<typename PROPERTY_TYPE>
bool_t PropertyEditorWidget::ShowVectorProperty(codeframe::PropertyBase* prop)
{
    auto propVector = dynamic_cast<codeframe::Property< std::vector<PROPERTY_TYPE> >*>(prop);

    if (nullptr != propVector)
    {
        std::vector<PROPERTY_TYPE>& internalVector = propVector->GetValue();

        PROPERTY_TYPE value = 0U;
        PROPERTY_TYPE valuePrew = 0U;
        static size_t index = 0;
        size_t indexPrew = 0;
        std::string vectorSizeIndexText = std::string("/") + std::to_string(internalVector.size()) + std::string(")");
        volatile ImVec2 vectorSizeIndexTextSize = ImGui::CalcTextSize(vectorSizeIndexText.c_str());
        float width = ImGui::GetColumnWidth() - 128.0F - vectorSizeIndexTextSize.x;
        vectorSizeIndexText +=  std::string("##vector_index");

        if (internalVector.size() > 0U)
        {
            if (index >= internalVector.size())
            {
                index = internalVector.size() - 1U;
            }
            value = valuePrew = internalVector[index];
            indexPrew = index;
            ImGui::PushItemWidth(width * 0.6F);
            ImGui::InputInt("=vector(", reinterpret_cast<int*>(&value), 1); ImGui::SameLine();
            ImGui::PopItemWidth();
            ImGui::PushItemWidth(width * 0.4F);
            ImGui::InputInt(vectorSizeIndexText.c_str(), reinterpret_cast<int*>(&index), 1); ImGui::SameLine();
            ImGui::PopItemWidth();
            internalVector[indexPrew] = value;
            if (ImGui::Button("-"))
            {
                internalVector.erase(internalVector.begin() + index);
                prop->PulseChanged();
            }
            ImGui::SameLine();

            if (valuePrew != value)
            {
                prop->PulseChanged();
            }
        }
        else
        {
            ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
            ImGui::PushItemWidth(width * 0.6F);
            ImGui::InputInt("=vector(##_value", reinterpret_cast<int*>(&index), 1); ImGui::SameLine();
            ImGui::PopItemWidth();
            ImGui::PushItemWidth(width * 0.4F);
            ImGui::InputInt(vectorSizeIndexText.c_str(), reinterpret_cast<int*>(&index), 1); ImGui::SameLine();
            ImGui::PopItemWidth();
            if (ImGui::Button("-"))
            {
            }
            ImGui::SameLine();
            ImGui::PopItemFlag();
            ImGui::PopStyleVar();
        }

        if (ImGui::Button("+"))
        {
            internalVector.insert(internalVector.begin() + index, 0U);
            prop->PulseChanged();
        }
        ImGui::SameLine();

        return true;
    }
    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<>
bool_t PropertyEditorWidget::ShowVectorProperty<RayData>(codeframe::PropertyBase* prop)
{
    auto propVector = dynamic_cast<codeframe::Property< std::vector<RayData> >*>(prop);

    if (nullptr != propVector)
    {
        std::vector<RayData>& internalVector = propVector->GetValue();

        float value;
        float valuePrew;
        static size_t index = 0;
        size_t indexPrew = 0;
        std::string vectorSizeIndexText = std::string("/") + std::to_string(internalVector.size()) + std::string(")");
        volatile ImVec2 vectorSizeIndexTextSize = ImGui::CalcTextSize(vectorSizeIndexText.c_str());
        float width = ImGui::GetColumnWidth() - 128.0F - vectorSizeIndexTextSize.x;
        vectorSizeIndexText +=  std::string("##vector_index");

        if (internalVector.size() > 0U)
        {
            if (index >= internalVector.size())
            {
                index = internalVector.size() - 1U;
            }
            value = valuePrew = internalVector[index].Distance;
            indexPrew = index;
            ImGui::PushItemWidth(width * 0.6F);
            ImGui::InputFloat("=vector(", &value, 0.1f); ImGui::SameLine();
            ImGui::PopItemWidth();
            ImGui::PushItemWidth(width * 0.4F);
            ImGui::InputInt(vectorSizeIndexText.c_str(), reinterpret_cast<int*>(&index), 1); ImGui::SameLine();
            ImGui::PopItemWidth();
            internalVector[indexPrew].Distance = value;
            if (ImGui::Button("-"))
            {
                internalVector.erase(internalVector.begin() + index);
                prop->PulseChanged();
            }
            ImGui::SameLine();

            if (valuePrew != value)
            {
                prop->PulseChanged();
            }
        }
        else
        {
            ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
            ImGui::PushItemWidth(width * 0.6F);
            ImGui::InputInt("=vector(##_value", reinterpret_cast<int*>(&index), 1); ImGui::SameLine();
            ImGui::PopItemWidth();
            ImGui::PushItemWidth(width * 0.4F);
            ImGui::InputInt(vectorSizeIndexText.c_str(), reinterpret_cast<int*>(&index), 1); ImGui::SameLine();
            ImGui::PopItemWidth();
            if (ImGui::Button("-"))
            {
            }
            ImGui::SameLine();
            ImGui::PopItemFlag();
            ImGui::PopStyleVar();
        }

        if (ImGui::Button("+"))
        {
            internalVector.insert(internalVector.begin() + index, RayData());
            prop->PulseChanged();
        }
        ImGui::SameLine();

        return true;
    }
    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<typename PROPERTY_TYPE>
bool_t PropertyEditorWidget::ShowVectorThrustHostProperty(codeframe::PropertyBase* prop)
{
    auto propThrustHostVector = dynamic_cast<codeframe::Property< thrust::host_vector<PROPERTY_TYPE> >*>(prop);

    if (nullptr != propThrustHostVector)
    {
        thrust::host_vector<PROPERTY_TYPE>& internalVector = propThrustHostVector->GetValue();

        PROPERTY_TYPE value = 0U;
        PROPERTY_TYPE valuePrew = 0U;
        static size_t index = 0;
        size_t indexPrew = 0;
        std::string vectorSizeIndexText = std::string("/") + std::to_string(internalVector.size()) + std::string(")");
        volatile ImVec2 vectorSizeIndexTextSize = ImGui::CalcTextSize(vectorSizeIndexText.c_str());
        float width = ImGui::GetColumnWidth() - 128.0F - vectorSizeIndexTextSize.x;
        vectorSizeIndexText +=  std::string("##thrust_vector_index");

        if (internalVector.size() > 0U)
        {
            if (index >= internalVector.size())
            {
                index = internalVector.size() - 1U;
            }
            value = valuePrew = internalVector[index];
            indexPrew = index;
            ImGui::PushItemWidth(width * 0.6F);
            ImGui::InputInt("=thrust(", reinterpret_cast<int*>(&value), 1); ImGui::SameLine();
            ImGui::PopItemWidth();
            ImGui::PushItemWidth(width * 0.4F);
            ImGui::InputInt(vectorSizeIndexText.c_str(), reinterpret_cast<int*>(&index), 1); ImGui::SameLine();
            ImGui::PopItemWidth();
            internalVector[indexPrew] = value;
            if (ImGui::Button("-"))
            {
                internalVector.erase(internalVector.begin() + index);
                prop->PulseChanged();
            }
            ImGui::SameLine();

            if (valuePrew != value)
            {
                prop->PulseChanged();
            }
        }
        else
        {
            ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
            ImGui::PushItemWidth(width * 0.6F);
            ImGui::InputInt("=thrust(##_value", reinterpret_cast<int*>(&index), 1); ImGui::SameLine();
            ImGui::PopItemWidth();
            ImGui::PushItemWidth(width * 0.4F);
            ImGui::InputInt(vectorSizeIndexText.c_str(), reinterpret_cast<int*>(&index), 1); ImGui::SameLine();
            ImGui::PopItemWidth();
            if (ImGui::Button("-"))
            {
            }
            ImGui::SameLine();
            ImGui::PopItemFlag();
            ImGui::PopStyleVar();
        }

        if (ImGui::Button("+"))
        {
            internalVector.insert(internalVector.begin() + index, value);
            prop->PulseChanged();
        }
        ImGui::SameLine();

        return true;
    }
    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<>
bool_t PropertyEditorWidget::ShowVectorThrustHostProperty<RayData>(codeframe::PropertyBase* prop)
{
    auto propThrustHostVector = dynamic_cast<codeframe::Property< thrust::host_vector<RayData> >*>(prop);

    if (nullptr != propThrustHostVector)
    {
        thrust::host_vector<RayData>& internalVector = propThrustHostVector->GetValue();

        float value;
        float valuePrew;
        static size_t index = 0;
        size_t indexPrew = 0;
        std::string vectorSizeIndexText = std::string("/") + std::to_string(internalVector.size()) + std::string(")");
        volatile ImVec2 vectorSizeIndexTextSize = ImGui::CalcTextSize(vectorSizeIndexText.c_str());
        float width = ImGui::GetColumnWidth() - 128.0F - vectorSizeIndexTextSize.x;
        vectorSizeIndexText +=  std::string("##thrust_vector_index");

        if (internalVector.size() > 0U)
        {
            if (index >= internalVector.size())
            {
                index = internalVector.size() - 1U;
            }
            value = valuePrew = internalVector[index].Distance;
            indexPrew = index;
            ImGui::PushItemWidth(width * 0.6F);
            ImGui::InputFloat("=thrust(", &value, 1); ImGui::SameLine();
            ImGui::PopItemWidth();
            ImGui::PushItemWidth(width * 0.4F);
            ImGui::InputInt(vectorSizeIndexText.c_str(), reinterpret_cast<int*>(&index), 1); ImGui::SameLine();
            ImGui::PopItemWidth();
            internalVector[indexPrew].Distance = value;
            if (ImGui::Button("-"))
            {
                internalVector.erase(internalVector.begin() + index);
                prop->PulseChanged();
            }
            ImGui::SameLine();

            if (valuePrew != value)
            {
                prop->PulseChanged();
            }
        }
        else
        {
            ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
            ImGui::PushItemWidth(width * 0.6F);
            ImGui::InputInt("=thrust(##_value", reinterpret_cast<int*>(&index), 1); ImGui::SameLine();
            ImGui::PopItemWidth();
            ImGui::PushItemWidth(width * 0.4F);
            ImGui::InputInt(vectorSizeIndexText.c_str(), reinterpret_cast<int*>(&index), 1); ImGui::SameLine();
            ImGui::PopItemWidth();
            if (ImGui::Button("-"))
            {
            }
            ImGui::SameLine();
            ImGui::PopItemFlag();
            ImGui::PopStyleVar();
        }

        if (ImGui::Button("+"))
        {
            internalVector.insert(internalVector.begin() + index, RayData());
            prop->PulseChanged();
        }
        ImGui::SameLine();

        return true;
    }
    return false;
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

        if (prop->IsReference())
        {
            ImGui::PushStyleColor(ImGuiCol_Text, 0xFFFF0000); ImGui::SameLine();
            ImGui::Text("link");
            ImGui::PopStyleColor();

            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(prop->Reference()->Path().c_str());
        }

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
                bool check = static_cast<bool>(*prop);
                ImGui::Checkbox("##value", &check);
                (*prop) = check;
                break;
            }
            case codeframe::KIND_NUMBER:
            {
                int value = static_cast<int>(*prop);
                ImGui::InputInt("##value", &value, 1);
                (*prop) = value;
                break;
            }
            case codeframe::KIND_NUMBERRANGE:
            {
                int value = static_cast<int>(*prop);
                ImGui::InputInt("##value", &value, 1);
                (*prop) = value;
                break;
            }
            case codeframe::KIND_REAL:
            {
                float value = static_cast<float>(*prop);
                ImGui::InputFloat("##value", &value, 1.0f);
                (*prop) = value;
                break;
            }
            case codeframe::KIND_TEXT:
            {
                char newText[32];
                memset(newText, 0, 32);
                std::string textValue = static_cast<std::string>(*prop);
                strncpy(newText, textValue.c_str(), textValue.length());
                ImGui::InputText("##value", newText, IM_ARRAYSIZE(newText));
                std::string editedText = std::string( newText );
                if ( textValue != editedText )
                {
                    (*prop) = editedText;
                }
                break;
            }
            case codeframe::KIND_ENUM:
            {
                std::string enumString = prop->Info().GetEnum();
                static ImGuiComboFlags flags = 0;

                if ( enumString.size() )
                {
                    std::vector<std::string> elems;
                    std::stringstream ss(enumString);
                    std::string currentValue;
                    while (std::getline(ss, currentValue, ',') )
                    {
                        elems.push_back( currentValue );
                    }

                    if (ImGui::BeginCombo("##Combo 1", elems[static_cast<unsigned int>(*prop)].c_str(), flags)) // The second parameter is the label previewed before opening the combo.
                    {
                        for (size_t n = 0; n < elems.size(); n++)
                        {
                            bool_t is_selected = (static_cast<unsigned int>(*prop) == n);
                            if (ImGui::Selectable(elems[n].c_str(), is_selected))
                            {
                                (*prop) = n;
                            }
                            if (is_selected)
                            {
                                ImGui::SetItemDefaultFocus();   // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
                            }
                        }
                        ImGui::EndCombo();
                    }
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
            case codeframe::KIND_VECTOR:
            {
                switch ( prop->Info().GetKind(DEPTH_INTERNAL_KIND) )
                {
                    case codeframe::KIND_NUMBER:
                    {
                        if (ShowVectorProperty<unsigned int>(prop) == false)
                        {
                            ShowVectorProperty<int>(prop);
                        }
                        break;
                    }
                    case codeframe::KIND_REAL:
                    {
                        if (ShowVectorProperty<float>(prop) == false)
                        {
                            ShowVectorProperty<double>(prop);
                        }
                        break;
                    }
                    case codeframe::KIND_RAY_DATA:
                    {
                        ShowVectorProperty<RayData>(prop);
                    }
                    default:
                    {
                    }
                }
                break;
            }
            case codeframe::KIND_VECTOR_THRUST_HOST:
            {
                switch ( prop->Info().GetKind(DEPTH_INTERNAL_KIND) )
                {
                    case codeframe::KIND_NUMBER:
                    {
                        if (ShowVectorThrustHostProperty<unsigned int>(prop) == false)
                        {
                            ShowVectorThrustHostProperty<int>(prop);
                        }
                        break;
                    }
                    case codeframe::KIND_REAL:
                    {
                        if (ShowVectorThrustHostProperty<float>(prop) == false)
                        {
                            ShowVectorThrustHostProperty<double>(prop);
                        }
                        break;
                    }
                    case codeframe::KIND_RAY_DATA:
                    {
                        ShowVectorThrustHostProperty<RayData>(prop);
                    }
                    default:
                    {
                    }
                }
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
