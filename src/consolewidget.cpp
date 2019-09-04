#include "consolewidget.hpp"

#include <cstring>

#include <TextUtilities.h>

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
void ConsoleWidget::Save( utilities::data::DataStorage& ds )
{
    m_History.Save( ds );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ConsoleWidget::Load( utilities::data::DataStorage& ds )
{
    m_History.Load( ds );
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
        utilities::text::trim(s);
        if (s[0])
        {
            std::string cmdString( s );
            m_parent.Script().RunString( cmdString );
            m_History.Push( cmdString );
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
int ConsoleWidget::TextEditCallbackStub(ImGuiInputTextCallbackData* data)
{
    ConsoleWidget* console = (ConsoleWidget*)data->UserData;
    return console->TextEditCallback(data);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
int ConsoleWidget::TextEditCallback(ImGuiInputTextCallbackData* data)
{
    switch (data->EventFlag)
    {
    case ImGuiInputTextFlags_CallbackCompletion:
        {
            /*
            // Locate beginning of current word
            const char* word_end = data->Buf + data->CursorPos;
            const char* word_start = word_end;
            while (word_start > data->Buf)
            {
                const char c = word_start[-1];
                if (c == ' ' || c == '\t' || c == ',' || c == ';')
                    break;
                word_start--;
            }

            // Build a list of candidates
            ImVector<const char*> candidates;
            for (int i = 0; i < m_Commands.Size; i++)
                if (Strnicmp(m_Commands[i], word_start, (int)(word_end-word_start)) == 0)
                    candidates.push_back(m_Commands[i]);

            if (candidates.Size == 0)
            {
                // No match
                AddLog("No match for \"%.*s\"!\n", (int)(word_end-word_start), word_start);
            }
            else if (candidates.Size == 1)
            {
                // Single match. Delete the beginning of the word and replace it entirely so we've got nice casing
                data->DeleteChars((int)(word_start-data->Buf), (int)(word_end-word_start));
                data->InsertChars(data->CursorPos, candidates[0]);
                data->InsertChars(data->CursorPos, " ");
            }
            else
            {
                // Multiple matches. Complete as much as we can, so inputing "C" will complete to "CL" and display "CLEAR" and "CLASSIFY"
                int match_len = (int)(word_end - word_start);
                for (;;)
                {
                    int c = 0;
                    bool all_candidates_matches = true;
                    for (int i = 0; i < candidates.Size && all_candidates_matches; i++)
                        if (i == 0)
                            c = toupper(candidates[i][match_len]);
                        else if (c == 0 || c != toupper(candidates[i][match_len]))
                            all_candidates_matches = false;
                    if (!all_candidates_matches)
                        break;
                    match_len++;
                }

                if (match_len > 0)
                {
                    data->DeleteChars((int)(word_start - data->Buf), (int)(word_end-word_start));
                    data->InsertChars(data->CursorPos, candidates[0], candidates[0] + match_len);
                }

                // List matches
                AddLog("Possible matches:\n");
                for (int i = 0; i < candidates.Size; i++)
                    AddLog("- %s\n", candidates[i]);
            }
            */
            break;
        }
        case ImGuiInputTextFlags_CallbackHistory:
        {
            std::string historyEntry("");

            if (data->EventKey == ImGuiKey_UpArrow)
            {
                historyEntry = m_History.PeekPrew();
            }
            else if (data->EventKey == ImGuiKey_DownArrow)
            {
                historyEntry = m_History.PeekNext();
            }

            if ( !historyEntry.empty() )
            {
                data->DeleteChars( 0, data->BufTextLen );
                data->InsertChars( 0, historyEntry.c_str() );
            }
        }
    }
    return 0;
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
