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
template<typename ValueType>
inline typename std::enable_if< std::is_integral< ValueType >::value, ValueType >::type
InputControlCreate(const std::string& name, const ValueType& valueBase)
{
    int value = static_cast<int>(valueBase);
    ImGui::InputInt(name.c_str(), &value, 1U);
    return value;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<typename ValueType>
inline typename std::enable_if< std::is_floating_point< ValueType >::value, ValueType >::type
InputControlCreate(const std::string& name, const ValueType& valueBase)
{
    float value = static_cast<float>(valueBase);
    ImGui::InputFloat(name.c_str(), &value, 0.1f);
    return value;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<typename ValueType>
inline typename std::enable_if< std::is_class< ValueType >::value, ValueType >::type
InputControlCreate(const std::string& name, const ValueType& valueBase)
{
    float value = valueBase.Distance;
    ImGui::InputFloat(name.c_str(), &value, 0.1f);
    return value;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<template<typename, typename> class ContainerType, typename ValueType, typename Allocator=std::allocator<ValueType>>
void ImgVectorEditor(codeframe::Property<ContainerType<ValueType, Allocator>>& propertyVectorObject)
{
    ContainerType<ValueType, Allocator>& internalVector = propertyVectorObject.GetValue();

    ValueType value(0);
    ValueType valuePrew(0);
    size_t index = propertyVectorObject.Index();
    size_t indexPrew = 0;
    std::string vectorSizeIndexText = std::string("/") + std::to_string(internalVector.size()) + std::string(")");
    volatile ImVec2 vectorSizeIndexTextSize = ImGui::CalcTextSize(vectorSizeIndexText.c_str());
    float width = ImGui::GetColumnWidth() - 128.0F - vectorSizeIndexTextSize.x;
    vectorSizeIndexText +=  std::string("##thrust_vector_index");

    if (internalVector.size() == 0U)
    {
        ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }

    if (internalVector.size() > 0U)
    {
        if (index >= internalVector.size())
        {
            index = internalVector.size() - 1U;
        }
        value = valuePrew = internalVector[index];
        indexPrew = index;
        ImGui::PushItemWidth(width * 0.6F);
        value = InputControlCreate<ValueType>("=thrust(", value); ImGui::SameLine();
        ImGui::PopItemWidth();
        ImGui::PushItemWidth(width * 0.4F);
        ImGui::InputInt(vectorSizeIndexText.c_str(), reinterpret_cast<int*>(&index), 1); ImGui::SameLine();
        ImGui::PopItemWidth();
        internalVector[indexPrew] = value;
        if (ImGui::Button("-"))
        {
            internalVector.erase(internalVector.begin() + index);
            propertyVectorObject.PulseChanged();
        }
        ImGui::SameLine();

        if (indexPrew != index)
        {
            propertyVectorObject.Index() = index;
        }

        if (valuePrew != value)
        {
            propertyVectorObject.PulseChanged();
        }
    }

    if (internalVector.size() == 0U)
    {
        ImGui::PopItemFlag();
        ImGui::PopStyleVar();
    }

    if (ImGui::Button("+"))
    {
        internalVector.insert(internalVector.begin() + index, value);
        propertyVectorObject.PulseChanged();
    }
    ImGui::SameLine();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<template<typename, typename> class ContainerType, typename ValueType, typename Allocator = std::allocator<ValueType>>
inline bool_t ShowVectorProperty(codeframe::PropertyBase* prop)
{
    auto propVector = dynamic_cast<codeframe::Property< ContainerType<ValueType, Allocator> >*>(prop);

    if (nullptr != propVector)
    {
        ImgVectorEditor<ContainerType, ValueType>(*propVector);
        return true;
    }
    return false;
}

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
                ImGui::SetTooltip("%s", prop->Reference()->Path().c_str());
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
                                (*prop) = (unsigned int)n;
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
                        if (ShowVectorProperty<std::vector, unsigned int>(prop) == false)
                        {
                            ShowVectorProperty<std::vector, int>(prop);
                        }
                        break;
                    }
                    case codeframe::KIND_REAL:
                    {
                        if (ShowVectorProperty<std::vector, float>(prop) == false)
                        {
                            ShowVectorProperty<std::vector, double>(prop);
                        }
                        break;
                    }
                    case codeframe::KIND_RAY_DATA:
                    {
                        ShowVectorProperty<std::vector, RayData>(prop);
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
                        if (ShowVectorProperty<thrust::host_vector, int>(prop) == false)
                        {
                            ShowVectorProperty<thrust::host_vector, unsigned int>(prop);
                        }
                        break;
                    }
                    case codeframe::KIND_REAL:
                    {
                        if (ShowVectorProperty<thrust::host_vector, float>(prop) == false)
                        {
                            ShowVectorProperty<thrust::host_vector, double>(prop);
                        }
                        break;
                    }
                    case codeframe::KIND_RAY_DATA:
                    {
                        ShowVectorProperty<thrust::host_vector, RayData>(prop);
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
