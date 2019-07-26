#include "consolewidget.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ConsoleWidget::ConsoleWidget( cSerializableInterface& parent ) :
    m_parent( parent ),
    m_ScrollToBottom( false )
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
    bool copy_to_clipboard = ImGui::Button("Copy");
    ImGui::SameLine();
    m_Filter.Draw("Filter", -100.0f);
    ImGui::Separator();
    const float footer_height_to_reserve = ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing(); // 1 separator, 1 input text
    ImGui::BeginChild("scrolling", ImVec2(0, -footer_height_to_reserve), false, ImGuiWindowFlags_HorizontalScrollbar|ImGuiWindowFlags_AlwaysAutoResize);

    if ( copy_to_clipboard == true )
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
            if ( m_Filter.PassFilter(line, line_end) )
            {
                ImGui::TextUnformatted(line, line_end);
            }
            line = line_end && line_end[1] ? line_end + 1 : NULL;
        }
    }
    else
    {
        ImGui::TextUnformatted( m_Buf.begin() );
    }

    if (m_ScrollToBottom)
    {
        ImGui::SetScrollHereY(1.0f);
        m_ScrollToBottom = false;
    }

    ImGui::Separator();
    ImGui::EndChild();

    // Command-line
    bool reclaim_focus = false;
    ImGui::PushItemWidth( ImGui::GetWindowWidth() );
    if (ImGui::InputText("Input", m_InputBuf, IM_ARRAYSIZE(m_InputBuf), ImGuiInputTextFlags_EnterReturnsTrue|ImGuiInputTextFlags_CallbackCompletion|ImGuiInputTextFlags_CallbackHistory, &TextEditCallbackStub, (void*)this))
    {
        char* s = m_InputBuf;
        Strtrim(s);
        if (s[0])
        {
            m_parent.Script().RunString( std::string( s ) );

            // Insert into history. First find match and delete it so it can be pushed to the back. This isn't trying to be smart or optimal.
            m_HistoryPos = -1;
            for (int i = m_History.Size-1; i >= 0; i--)
                if (Stricmp(m_History[i], s) == 0)
                {
                    free(m_History[i]);
                    m_History.erase(m_History.begin() + i);
                    break;
                }
            m_History.push_back(Strdup(s));
        }
        strcpy(s, "");
        reclaim_focus = true;
    }

    // Auto-focus on window apparition
    ImGui::SetItemDefaultFocus();
    if (reclaim_focus)
    {
        ImGui::SetKeyboardFocusHere(-1); // Auto focus previous widget
    }

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
    {
        if (m_Buf[old_size] == '\n')
        {
            m_LineOffsets.push_back(old_size);
        }
    }
    m_ScrollToBottom = true;
}
