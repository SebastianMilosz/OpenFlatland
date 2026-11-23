#include "ann_viewer_widget.hpp"
#include "colorize_number.hpp"

#include <MathUtilities.h>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
AnnViewerWidget::AnnViewerWidget()
{
    m_renderStates.blendMode = sf::BlendMode(sf::BlendMode::Factor::One, sf::BlendMode::Factor::OneMinusSrcAlpha);
    (void)m_displayTexture.resize({SCREEN_WIDTH, SCREEN_HEIGHT});
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void AnnViewerWidget::SetObject( smart_ptr<codeframe::ObjectNode> obj )
{
    m_objEntity = smart_dynamic_pointer_cast<Entity>(obj);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void AnnViewerWidget::Draw( const char* title, bool* p_open )
{
    ImGui::SetNextWindowSize(ImVec2(430, 450), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(title, p_open))
    {
        ImGui::End();
        return;
    }

    ImGui::PushItemWidth(MENU_LEFT_OFFSET - 30);
    ImGui::Text("Node = "); ImGui::SameLine();
    ImGui::InputInt("##Node=", &m_selectedNode, 1);
    ImGui::PopItemWidth();

    m_displayTexture.clear();

    if (smart_ptr_isValid(m_objEntity))
    {
        ArtificialNeuronEngine& engine = m_objEntity->GetEngine();
        DrawableSpikingNeuralNetwork& neuronPool = dynamic_cast<DrawableSpikingNeuralNetwork&>(engine.GetPool());

        neuronPool.Select(m_selectedNode);

        // Show the hovered node, fall back to the selected one
        int32_t infoNode = neuronPool.GetHovered();
        const char* infoSource = "Hovered";

        if (infoNode == SelectableElement::NO_NODE)
        {
            infoSource = "Selected";
            infoNode = m_selectedNode;
        }

        ImGui::BeginChild("OuterRegion", ImVec2(MENU_LEFT_OFFSET, 300), false);

        if (neuronPool.IsValidNode(infoNode))
        {
            ImGui::TextUnformatted(infoSource);
            for (const auto& infoLine : neuronPool.GetBlockInfo(infoNode))
            {
                ImGui::TextUnformatted(std::get<0>(infoLine).c_str());
                ImGui::SameLine();
                ImGui::TextUnformatted(std::get<1>(infoLine).c_str());
            }
        }
        else
        {
            ImGui::TextUnformatted("Hover a node to inspect it,");
            ImGui::TextUnformatted("click to pin the selection.");
        }

        ImGui::EndChild();

        ImGui::Spacing();

        m_displayTexture.draw(neuronPool, m_renderStates);

        // Center vision screen inside imgui widget
        m_cursorPos = ImGui::GetWindowSize();
        m_cursorPos.x = (m_cursorPos.x - MENU_LEFT_OFFSET - SCREEN_WIDTH) * 0.5f + MENU_LEFT_OFFSET;
        m_cursorPos.y = (m_cursorPos.y - SCREEN_HEIGHT) * 0.5f;

        if (m_cursorPos.x < MENU_LEFT_OFFSET)
        {
            m_cursorPos.x = MENU_LEFT_OFFSET + 20U;
        }

        ImGui::SetCursorPos(m_cursorPos);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));

        m_displayTexture.display();
        // RenderTexture overload flips V, render textures are stored upside down
        ImGui::Image(m_displayTexture);

        // --- Handle Mouse Interactions Over the SNN Viewport ---
        if (ImGui::IsItemHovered())
        {
            // Mouse position relative to the top left corner of the texture
            const ImVec2 imageMin = ImGui::GetItemRectMin();
            const ImVec2 mousePos = ImGui::GetMousePos();
            const sf::Vector2u textureSize = m_displayTexture.getSize();

            neuronPool.OnHoverPixels(mousePos.x - imageMin.x,
                                     mousePos.y - imageMin.y,
                                     static_cast<float>(textureSize.x),
                                     static_cast<float>(textureSize.y));

            // Click pins the hovered node as the selection
            const int32_t hovered = neuronPool.GetHovered();
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && hovered != SelectableElement::NO_NODE)
            {
                m_selectedNode = hovered;
            }
        }
        else
        {
            // Reset hover state if mouse leaves the canvas
            neuronPool.OnHover(SelectableElement::NO_NODE);
        }

        ImGui::PopStyleVar();
    }

    ImGui::End();
}
