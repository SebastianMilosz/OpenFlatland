#include "consolewidget.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ConsoleWidget::ConsoleWidget()
{
    memset(m_InputBuf, 0, sizeof(m_InputBuf));
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ConsoleWidget::~ConsoleWidget()
{
    //dtor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ConsoleWidget::OnLogMessage(const std::string& timestamp, const std::string& title, const std::string& msg, int type)
{
    AddLog( "[%s] : [%s] : [%d] : %s\n", timestamp.c_str(), title.c_str(), type, msg.c_str() );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ConsoleWidget::Draw( const char* title, bool* p_open )
{
    ImGui::SetNextWindowSize(ImVec2(500,400), ImGuiCond_FirstUseEver);
    ImGui::Begin(title, p_open);
    if (ImGui::Button("Clear")) Clear();
    ImGui::SameLine();
    bool copy = ImGui::Button("Copy");
    ImGui::SameLine();
    m_Filter.Draw("Filter", -100.0f);
    ImGui::Separator();
    ImGui::BeginChild("scrolling", ImVec2(0,0), false, ImGuiWindowFlags_HorizontalScrollbar);
    if (copy)
    {
        ImGui::LogToClipboard();
    }

    if (m_Filter.IsActive())
    {
        const char* buf_begin = m_Buf.begin();
        const char* line = buf_begin;
        for (int line_no = 0; line != NULL; line_no++)
        {
            const char* line_end = (line_no < m_LineOffsets.Size) ? buf_begin + m_LineOffsets[line_no] : NULL;
            if (m_Filter.PassFilter(line, line_end))
                ImGui::TextUnformatted(line, line_end);
            line = line_end && line_end[1] ? line_end + 1 : NULL;
        }
    }
    else
    {
        ImGui::TextUnformatted(m_Buf.begin());
    }

    ImGui::Separator();

    // Command-line
    bool reclaim_focus = false;
    if (ImGui::InputText("Input", m_InputBuf, IM_ARRAYSIZE(m_InputBuf), ImGuiInputTextFlags_EnterReturnsTrue|ImGuiInputTextFlags_CallbackCompletion|ImGuiInputTextFlags_CallbackHistory, &TextEditCallbackStub, (void*)this))
    {
        char* s = m_InputBuf;
        Strtrim(s);
        if (s[0])
        {

        }
        strcpy(s, "");
        reclaim_focus = true;
    }

    if (m_ScrollToBottom)
    {
        ImGui::SetScrollHere(1.0f);
    }
    m_ScrollToBottom = false;
    ImGui::EndChild();
    ImGui::End();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ConsoleWidget::Clear()
{
    m_Buf.clear();
    m_LineOffsets.clear();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ConsoleWidget::AddLog(const char* fmt, ...)
{
    int old_size = m_Buf.size();
    va_list args;
    va_start(args, fmt);
    m_Buf.appendfv(fmt, args);
    va_end(args);
    for (int new_size = m_Buf.size(); old_size < new_size; old_size++)
        if (m_Buf[old_size] == '\n')
            m_LineOffsets.push_back(old_size);
    m_ScrollToBottom = true;
}
