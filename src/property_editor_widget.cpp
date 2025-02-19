#include "property_editor_widget.hpp"

#ifdef USE_RAYDATA_EXT_TYPE
#include <ray_data.hpp>
#endif

#ifdef USE_THRUST_EXT_TYPE
#include <thrust/device_vector.h>
#endif

#include <imgui_internal.h> // Currently imgui dosn't have disable/enable control feature
#include <MathUtilities.h>

namespace ImGui
{
    enum ImGuiTristateFlags
    {
        TristateL = 0x00,
        TristateX = 0x01,
        TristateY = 0x02,
        TristateH = 0x03
    };

    bool CheckBoxTristate(const char* label, uint32_t* v_tristate)
    {
        bool ret;
        if ((*v_tristate == TristateX) || (*v_tristate == TristateY))
        {
            ImGui::PushItemFlag(ImGuiItemFlags_MixedValue, true);
            bool b = false;
            ret = ImGui::Checkbox(label, &b);
            if (ret)
            {
                *v_tristate = TristateH;
            }
            ImGui::PopItemFlag();
        }
        else
        {
            bool b = (*v_tristate != TristateL);
            ret = ImGui::Checkbox(label, &b);

            if (ret)
            {
                if (*v_tristate == TristateH)
                {
                    *v_tristate = TristateL;
                }
                else
                {
                    *v_tristate = TristateX;
                }
            }
        }
        return ret;
    }
}

using namespace codeframe;

class ImGuiDisabled
{
    public:
        ImGuiDisabled(const bool_t disabled = false) :
            m_disabled(disabled)
        {
            if (m_disabled)
            {
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
            }
        }

        ~ImGuiDisabled()
        {
            if (m_disabled)
            {
                ImGui::PopItemFlag();
                ImGui::PopStyleVar();
            }
        }
    private:
        const bool_t m_disabled;
};

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<typename ValueType>
inline typename std::enable_if< std::is_integral< ValueType >::value, ValueType >::type
InputControlCreate(const std::string& name, const ValueType& valueBase, const bool_t readOnly, uint32_t options = 0U)
{
    ImGuiDisabled disableGui(readOnly);
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
InputControlCreate(const std::string& name, const ValueType& valueBase, const bool_t readOnly, uint32_t options = 0U)
{
    ImGuiDisabled disableGui(readOnly);
    float value = static_cast<float>(valueBase);
    ImGui::InputFloat(name.c_str(), &value, 0.1f, 0.1f, "%.2E");

    // Check if the input field is hovered
    if (options)
    {
        if (ImGui::IsItemHovered())
        {
            ImGui::BeginTooltip();

            static uint8_t i = 0U;
            static std::array<float, 20U> values = {0U};
            values[i++ % 20U] = value;
            if (i > 20U) i = 0U;

            // Display the graph
            ImGui::PlotLines("##Values", values.data(), values.size());
            ImGui::EndTooltip();
        }
    }

    return value;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<typename ValueType>
inline typename std::enable_if< std::is_class< ValueType >::value, ValueType >::type
InputControlCreate(const std::string& name, const ValueType& valueBase, const bool_t readOnly, uint32_t options = 0U)
{
    ImGuiDisabled disableGui(readOnly);
    float value = valueBase.Distance;
    ImGui::InputFloat(name.c_str(), &value, 0.1f);
    return value;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
inline bool_t ButtonCreate(const std::string& name, const bool_t readOnly)
{
    ImGuiDisabled disableGui(readOnly);
    return ImGui::Button(name.c_str());
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<template<typename, typename> class ContainerType, typename ValueType, typename Allocator=std::allocator<ValueType>>
void ImgVectorEditor(codeframe::Property<ContainerType<ValueType, Allocator>>& propertyVectorObject, uint32_t options = 0U)
{
    auto& internalVector            = propertyVectorObject.GetValue();
    const auto& internalConstVector = propertyVectorObject.GetConstValue();
    bool_t readOnly = propertyVectorObject.IsValueReadOnly() || propertyVectorObject.Info().GetGuiEnable();

    ValueType value(0);
    ValueType valuePrew(0);
    size_t index = propertyVectorObject.Index();
    size_t indexPrew = 0;
    std::string vectorSizeIndexText = std::string("/") + std::to_string(internalConstVector.size()) + std::string(")");
    volatile ImVec2 vectorSizeIndexTextSize = ImGui::CalcTextSize(vectorSizeIndexText.c_str());
    float width = ImGui::GetColumnWidth() - 128.0F - vectorSizeIndexTextSize.x;
    vectorSizeIndexText +=  std::string("##thrust_vector_index");

    if (internalConstVector.size() > 0U)
    {
        if (index >= internalConstVector.size())
        {
            index = internalConstVector.size() - 1U;
        }
        value = valuePrew = internalConstVector[index];
        indexPrew = index;
        ImGui::PushItemWidth(width * 0.6F);
        value = InputControlCreate<ValueType>("=thrust(", value, readOnly, options); ImGui::SameLine();
        ImGui::PopItemWidth();

        ImGui::PushItemWidth(width * 0.4F);
        index = InputControlCreate(vectorSizeIndexText, index, false, options); ImGui::SameLine();
        ImGui::PopItemWidth();

        if (readOnly == false)
        {
            if (valuePrew != value)
            {
                internalVector[indexPrew] = value;
                propertyVectorObject.PulseChanged();
            }
        }

        if (ButtonCreate("-", readOnly))
        {
            internalVector.erase(internalVector.begin() + index);
            propertyVectorObject.PulseChanged();
        }

        ImGui::SameLine();
    }

    if (indexPrew != index)
    {
        propertyVectorObject.Index() = index;
    }

    if (ButtonCreate("+", readOnly))
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
template<template<typename> class ContainerType, typename ValueType>
void ImgPoint2dEditor(const std::string& label1, const std::string& label2, codeframe::Property< codeframe::Point2D<ValueType> >& propertyPoint2Object)
{
    auto& internalPoint = propertyPoint2Object.GetValue();

    ValueType valueX(internalPoint.X());
    ValueType valueXPrew(valueX);
    ValueType valueY(internalPoint.Y());
    ValueType valueYPrew(valueY);

    float widthText = ImGui::CalcTextSize(label1.c_str()).x + ImGui::CalcTextSize(label2.c_str()).x;
    float width = ImGui::GetColumnWidth() - widthText - 40;

    ImGui::PushItemWidth(width * 0.5f);
    ImGui::Text( label1.c_str() ); ImGui::SameLine();
    valueX = InputControlCreate<ValueType>("##valueX(", valueX, false); ImGui::SameLine();
    ImGui::PopItemWidth();

    ImGui::PushItemWidth(width * 0.5f);
    ImGui::Text( label2.c_str() ); ImGui::SameLine();
    valueY = InputControlCreate<ValueType>("##valueY(", valueY, false); ImGui::SameLine();
    ImGui::PopItemWidth();

    if (valueX != valueXPrew || valueY != valueYPrew)
    {
        internalPoint.Set(valueX, valueY);
        propertyPoint2Object.PulseChanged();
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<template<typename, typename> class ContainerType, typename ValueType, typename Allocator = std::allocator<ValueType>>
inline bool_t ShowVectorProperty(codeframe::PropertyBase* prop, uint32_t options = 0U)
{
    auto propVector = dynamic_cast<codeframe::Property< ContainerType<ValueType, Allocator> >*>(prop);

    if (nullptr != propVector)
    {
        ImgVectorEditor<ContainerType, ValueType>(*propVector, options);
        return true;
    }
    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<template<typename> class ContainerType, typename ValueType>
inline bool_t ShowPoint2dProperty(codeframe::PropertyBase* prop)
{
    auto propVector = dynamic_cast<codeframe::Property< ContainerType<ValueType> >*>(prop);

    if (nullptr != propVector)
    {
        ImgPoint2dEditor<ContainerType, ValueType>("W:", "H:", *propVector);
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
    ImGui::SameLine();

    if (ImGui::TextLink(valueChangeGraphEnable ? "(G)" : " G "))
    {
        if (valueChangeGraphEnable)
            valueChangeGraphEnable = false;
        else
            valueChangeGraphEnable = true;
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
codeframe::VariantValue PropertyEditorWidget::ShowSelectParameterDialog(smart_ptr<codeframe::ObjectNode> obj,
                                                                                 codeframe::eType typeFilter)
{
    codeframe::VariantValue retValue = codeframe::VariantValue();

    uint32_t uid(obj->Identity().GetUId());

    // Take object pointer as unique id
    ImGui::PushID(uid);

    std::string objectName = obj->Identity().ObjectName();
    bool node_open = false;

    if (obj->Role() == eBuildRole::CONTAINER)
    {
        node_open = ImGui::TreeNodeEx("Object", ImGuiTreeNodeFlags_AllowItemOverlap, "%s[%d]", objectName.c_str(), obj->Count());
    }
    else
    {
        node_open = ImGui::TreeNodeEx("Object", ImGuiTreeNodeFlags_AllowItemOverlap, "%s", objectName.c_str());
    }

    if (node_open == true)
    {
        // Iterate through properties in object
        for (auto it = obj->PropertyList().begin(); it != obj->PropertyList().end(); ++it)
        {
            codeframe::PropertyBase* iser = *it;

            if (iser)
            {
                if ((typeFilter == codeframe::eType::TYPE_NON) || (iser->Type() == typeFilter))
                {
                    if (ImGui::TextLink(iser->Name().c_str()))
                    {
                        retValue = codeframe::VariantValue("href", iser->Path());
                    }
                }
            }
        }

        if (!retValue.IsValid())
        {
            // Iterate through childs in the object
            for (auto it = obj->ChildList().begin(); it != obj->ChildList().end(); ++it)
            {
                auto childObject = *it;

                codeframe::VariantValue value = ShowSelectParameterDialog(childObject, typeFilter);
                if (value.IsValid())
                {
                    retValue = value;
                    break;
                }
            }
        }

        ImGui::TreePop();
    }

    ImGui::PopID();

    return retValue;
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
        node_open = ImGui::TreeNodeEx( "Object", ImGuiTreeNodeFlags_AllowItemOverlap, "%s[%d]", objectName.c_str(), obj->Count() );

        const std::vector< std::string >& classSet = obj->ClassSet();

        if (classSet.size())
        {
            ImGui::SameLine();
            if (ImGui::Button("+"))
            {
                ImGui::OpenPopup("ElementList");
            }

            if (ImGui::BeginPopup("ElementList"))
            {
                for(const auto& className : classSet)
                {
                    if (ImGui::Button(className.c_str()))
                    {
                        ImGui::OpenPopup(className.c_str());
                    }

                    if (ImGui::BeginPopupModal(className.c_str()))
                    {
                        codeframe::VariantValue value = ShowSelectParameterDialog(m_obj, codeframe::eType::TYPE_VECTOR);

                        if (value.IsValid())
                        {
                            std::string objectName = className + std::string("Obj");
                            obj->Create(className, objectName, std::vector<codeframe::VariantValue>{value});
                            ImGui::CloseCurrentPopup();
                        }

                        if (ImGui::Button("cancel"))
                        {
                            ImGui::CloseCurrentPopup();
                        }

                        ImGui::EndPopup();
                    }
                }

                ImGui::EndPopup();
            }
        }

        if (obj->BuildType() == eBuildType::DYNAMIC)
        {
            ImGui::SameLine();
            if (ImGui::Button("-"))
            {

            }
        }
    }
    else if (obj->Role() == eBuildRole::OBJECT)
    {
        node_open = ImGui::TreeNodeEx( "Object", ImGuiTreeNodeFlags_AllowItemOverlap, "%s", objectName.c_str() );

        if (obj->BuildType() == eBuildType::DYNAMIC)
        {
            ImGui::SameLine();
            if (ImGui::Button("-"))
            {
                smart_ptr<ObjectNode> parentNode = obj->Parent()->GetNode();
                if (smart_ptr_isValid(parentNode) && parentNode->Role() == eBuildRole::CONTAINER)
                {
                    smart_ptr<ObjectContainer> containerNode = smart_dynamic_pointer_cast<ObjectContainer>(parentNode);

                    if (smart_ptr_isValid(containerNode))
                    {
                        containerNode->Dispose(obj);
                    }
                }
            }
        }
    }

    if (obj->Role() != eBuildRole::HIDDEN)
    {
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
            case codeframe::KIND_BIT_FIELD_2S:
            {
                uint8_t visibleRange = prop->Info().GetVisibleRange();
                uint32_t flagsIn = static_cast<unsigned int>(*prop);
                uint32_t flagsOut = 0U;

                for (uint8_t n = 0U; n < visibleRange; n++ )
                {
                    uint32_t flags = (flagsIn >> n*2U) & 0x03U;

                    std::string checkboxNameFinal = std::string("##") + prop->Name() + utilities::math::IntToStr(n);

                    if ( ImGui::CheckBoxTristate( checkboxNameFinal.c_str(), &flags ) )
                    {
                    }
                    ImGui::SameLine();

                    flagsOut |= (flags << n*2U);
                }

                *prop = flagsOut;

                break;
            }
            case codeframe::KIND_VECTOR:
            {
                switch ( prop->Info().GetKind(DEPTH_INTERNAL_KIND) )
                {
                    case codeframe::KIND_NUMBER:
                    {
                        if (ShowVectorProperty<std::vector, unsigned int>(prop, valueChangeGraphEnable) == false)
                        {
                            ShowVectorProperty<std::vector, int>(prop, valueChangeGraphEnable);
                        }
                        break;
                    }
                    case codeframe::KIND_REAL:
                    {
                        if (ShowVectorProperty<std::vector, float>(prop, valueChangeGraphEnable) == false)
                        {
                            ShowVectorProperty<std::vector, double>(prop, valueChangeGraphEnable);
                        }
                        break;
                    }
#ifdef USE_RAYDATA_EXT_TYPE
                    case codeframe::KIND_RAY_DATA:
                    {
                        ShowVectorProperty<std::vector, RayData>(prop, valueChangeGraphEnable);
                    }
#endif
                    default:
                    {
                    }
                }
                break;
            }
#ifdef USE_THRUST_EXT_TYPE
            case codeframe::KIND_VECTOR_THRUST_HOST:
            {
                switch ( prop->Info().GetKind(DEPTH_INTERNAL_KIND) )
                {
                    case codeframe::KIND_NUMBER:
                    {
                        if (ShowVectorProperty<thrust::host_vector, int>(prop, valueChangeGraphEnable) == false)
                        {
                            ShowVectorProperty<thrust::host_vector, unsigned int>(prop, valueChangeGraphEnable);
                        }
                        break;
                    }
                    case codeframe::KIND_REAL:
                    {
                        if (ShowVectorProperty<thrust::host_vector, float>(prop, valueChangeGraphEnable) == false)
                        {
                            ShowVectorProperty<thrust::host_vector, double>(prop, valueChangeGraphEnable);
                        }
                        break;
                    }
                    case codeframe::KIND_RAY_DATA:
                    {
                        ShowVectorProperty<thrust::host_vector, RayData>(prop, valueChangeGraphEnable);
                    }
                    default:
                    {
                    }
                }
                break;
            }
#endif
            case KIND_2DPOINT:
            {
                switch ( prop->Info().GetKind(DEPTH_INTERNAL_KIND) )
                {
                    case codeframe::KIND_NUMBER:
                    {
                        if (ShowPoint2dProperty<codeframe::Point2D, int>(prop) == false)
                        {
                            ShowPoint2dProperty<codeframe::Point2D, unsigned int>(prop);
                        }

                        break;
                    }
                    case codeframe::KIND_REAL:
                    {
                        if (ShowPoint2dProperty<codeframe::Point2D, float>(prop) == false)
                        {
                            ShowPoint2dProperty<codeframe::Point2D, double>(prop);
                        }

                        break;
                    }
                    default:
                    {
                    }
                }

                break;
            }
            case KIND_EVENT:
            {
                if (ImGui::Button((prop->Name()+"##").c_str()))
                {
                    uint32_t eventValue = prop->Info().GetEventValue();
                    (*prop) = eventValue;
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
