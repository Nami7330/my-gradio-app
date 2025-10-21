import gradio as gr
import os
from ai_chatbot_calculator import chatbot_calculator_logic

chat_history = []

def chat(user_input, history):
    """Return updated Chatbot-style history (list of (user, bot) tuples)."""
    response = chatbot_calculator_logic(user_input)
    history = history + [(user_input, response)]
    return history, ""   # (new chatbot value, clear textbox)

def clear_chat():
    return [], ""   # clear chatbot + textbox

BASE_CSS = """
html, body {
    margin: 0 !important;
    padding: 0 !important;
    box-sizing: border-box !important;
    width: 100vw !important;
    overflow-x: hidden !important;
}
/* --- rest of your CSS, unchanged --- */
#main-row {
    display: flex;
    width: 100%;
        transition: all 0.4s ease;
        align-items: stretch;
        gap: 20px;
    }
    #chat-area-wrapper {
        height: 60vh;
        padding-right: 8px;
        margin-bottom: 10px;
        background-color: white;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
        display: flex;
        flex-direction: column;
    }
    #chat-area { height: 100%; flex: 1 1 0%; }

    #sidebar {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 12px;
        max-width: 280px;
        min-width: 240px;
        transition: transform 0.4s ease;
        position: relative;
        margin-left: 10px;
        flex-shrink: 0;
        box-sizing: border-box;
        height: calc(100vh - 40px);
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    #examples-list {
        flex-grow: 1;
        max-height: 300px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 6px;
        border-radius: 8px;
        border: 1px solid #e3dbed;
        background: #faf7ff;
        scrollbar-width: thin;
        scrollbar-color: #c09ddf #f6ebfc;
        min-height: 80px;
    }
    #examples-list::-webkit-scrollbar {
        width: 8px;
        border-radius: 8px;
        background: #f6ebfc;
    }
    #examples-list::-webkit-scrollbar-thumb {
        background: #c09ddf;
        border-radius: 8px;
    }
    .hide-sidebar {
        transform: translateX(-100%);
        width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;
    }
    #toggle-btn {
        position: absolute;
        top: 0,
        right: 0;
        font-size: 20px;
        padding: 6px 10px;
        background-color: #c9a6dd;
        color: white;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        transition: background-color 0.3s ease, box-shadow 0.3s ease;
    }
    #toggle-btn:hover {
        background-color: #b28fcf;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    #chat-column { transition: all 0.4s ease; display: flex; flex-direction: column; height: 100%; }
    .expanded-chat { flex-grow: 1; }
    .input-wrapper {
        background-color: white;
        padding: 10px;
        border-radius: 12px;
        display: flex;
        gap: 10px;
        align-items: center;
        margin-top: 10px;
        box-shadow: 0 0 0 2px #e2d4e5;
    }
    .user-bubble {
        background-color: #d4bbf0;
        color: black;
        border-radius: 16px 16px 0 16px;
        align-self: flex-end;
    }
    .bot-bubble {
        background-color: #f2f2f2;
        color: black;
        border-radius: 16px 16px 16px 0;
        align-self: flex-start;
    }

    /* Apply the same styles to Gradio Chatbot bubbles */
    /* This targets common Gradio structures across 3.x and 4.x */
    #chat-area .message.user,
    #chat-area .message.user .bubble,
    #chat-area .wrap .message.user,
    #chat-area .bubble.user {
        background-color: #d4bbf0 !important;
        color: black !important;
        border-radius: 16px 16px 0 16px !important;
        align-self: flex-end !important;
    }
    #chat-area .message.bot,
    #chat-area .message.bot .bubble,
    #chat-area .wrap .message.bot,
    #chat-area .bubble.bot {
        background-color: #f2f2f2 !important;
        color: black !important;
        border-radius: 16px 16px 16px 0 !important;
        align-self: flex-start !important;
    }

    .submit-btn button {
        background-color: white !important;
        color: #f5f5f5 !important;
        border: 2px solid #7e4f9e !important;
        padding: 12px 20px !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        font-size: 1.05rem !important;
        min-height: 48px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.25s ease-in-out !important;
        margin: 0;
    }
    .submit-btn button:hover {
        box-shadow: 0 4px 12px rgba(126, 79, 158, 0.35) !important;
        border-color: #6a3b8c !important;
    }
    .heading {
        text-align: center;
        font-size: 26px;
        padding-bottom: 12px;
        border-bottom: 2px solid #eee;
        margin-bottom: 10px;
    }
    .styler { background: transparent !important; }

    /* --- Responsive Mobile Styles --- */
    @media (max-width: 700px) {

        #main-row {
            flex-direction: column;
            gap: 0;
            height: 100vh;
            min-height: 100vh;
            max-height: 100vh;
            align-items: stretch;
        }

        #main-row, #chat-column, #sidebar {
            width: 100vw !important;
            min-width: 0 !important;
            max-width: 100vw !important;
            margin: 0 !important;
            /* padding: 0 !important;  -- keep this to reset misc paddings, but override below */
            box-sizing: border-box;
            overflow-x: hidden;
        }

        html, body {
            /* Prevent horizontal scroll and cutoff on mobile */
            overflow-x: hidden !important;
            padding: 0 !important;
            margin: 0 !important;
            width: 100vw !important;
            box-sizing: border-box !important;
        }

        #main-row,
        #chat-column,
        #sidebar {
            width: 100% !important;
            min-width: 0 !important;
            max-width: 100% !important;
            margin-left: auto !important;
            margin-right: auto !important;
            left: 0 !important;
            right: 0 !important;
            box-sizing: border-box !important;
        }

        #sidebar {
            /* Set sidebar to 44% of viewport for more room on mobile */
            height: 30vh !important;
            border-bottom: 2px solid #eee;
            padding: 18px 16px 16px 16px !important; /* Ensure actual padding appears */
            box-sizing: border-box;
            display: flex !important;
            flex-direction: column !important;
        }
        #examples-list {
            flex: 1 1 0%;
            max-width: 100vw;
            width: 100vw;
            min-height: 0;
            max-height: none;
            font-size: 0.96rem;
            border-radius: 0;
            box-sizing: border-box;
            overflow-y: auto !important;    /* Ensure this stays */
            /* Remove min-height so content can scroll immediately */
            min-height: 0 !important;
        }
        #chat-column {
            /* Set chat-column to 56% of viewport */
            height: 56vh !important;
            display: flex;
            flex-direction: column;
        }
        #chat-area-wrapper {
            flex: 1 1 0%;
            height: calc(56vh - 62px);
            min-height: 0;
            max-height: none;
            padding: 6px 10px;
            margin-bottom: 0;
            border-radius: 0;
            box-shadow: none;
            background: white;
            display: flex;
            flex-direction: column;
        }
        #chat-area {
            height: 100%;
            max-height: none;
            flex: 1 1 0%;
        }
        .input-wrapper {
            flex-direction: column;
            gap: 6px;
            padding: 6px 0 0 0;
            border-radius: 0;
            margin-top: 0;
            margin-bottom: 0;
            min-height: 56px;
        }
        html, body {
            width: 100% !important;
            min-width: 0 !important;
            max-width: 100% !important;
            overflow-x: hidden !important;
            padding: 0 !important;
            margin: 0 !important;
            box-sizing: border-box !important;
        }
        #main-row, #chat-column, #sidebar {
            width: 100% !important;
            min-width: 0 !important;
            max-width: 100% !important;
            margin: 0 auto !important;
            left: auto !important;
            right: auto !important;
            padding: 0 !important;
            box-sizing: border-box !important;
            overflow-x: hidden !important;
            border-radius: 0 !important;
        }
        #chat-area-wrapper, #examples-list {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            margin: 0 !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
            box-sizing: border-box !important;
            overflow-x: hidden !important;
            border-radius: 0 !important;
        }
        .input-wrapper {
            width: 100% !important;
            min-width: 0 !important;
            max-width: 100vw !important;
            box-sizing: border-box !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
            align-items: stretch !important;
        }
        .input-wrapper .gr-textbox,
        .input-wrapper .gr-button {
            flex: 1 1 0%;
            width: 100% !important;
            min-width: 0 !important;
            box-sizing: border-box !important;
        }
        .input-wrapper .gr-button {
            /* Avoid button being super tall, but keep full width */
            max-width: 100% !important;
        }
        #sidebar, #sidebar * {
            text-align: center !important;
            align-items: center !important;
            justify-content: center !important; 
        }
        #sidebar {
            padding-left: 10px !important;
            padding-right: 10px !important;
            padding-top: 12px !important;
            padding-bottom: 12px !important;
        }
    }
"""

RESPONSIVE_CHAT_CONTAINER_CSS = """
/* --- Responsive Chat Area and Input Row for All Devices --- */

/* Chat column uses full available flex space */
#chat-column {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0; /* Allows descendant flex items to shrink if needed */
}

/* Make chat area wrapper flexible, not fixed height, on desktop */
#chat-area-wrapper {
    flex: 1 1 auto;
    min-height: 0;
    max-height: 100%;
    display: flex;
    flex-direction: column;
    padding: 20px;
    background: white;
    border-radius: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    margin-bottom: 10px;
}

/* Chatbot area fills most of the available vertical space */
#chat-area {
    flex: 1 1 0%;
    min-height: 0;
    max-height: 100%;
    overflow-y: auto;
}

.input-wrapper {
    display: flex;
    flex-direction: row;
    align-items: stretch;
    gap: 10px;
    padding: 10px;
    border-radius: 12px;
    box-shadow: 0 0 0 2px #e2d4e5;
    background: white;
    margin-top: 0;
    margin-bottom: 0;
}

/* Make input row and its components flexible on all screens */
.input-wrapper .gr-textbox {
    flex: 1 1 0%;
    min-width: 0;
}

.input-wrapper .gr-button {
    width: auto;
    min-width: 90px;
    flex-shrink: 0;
}

/* Responsive adjustments for chat/input at small widths */
@media (max-width: 700px) {
#chat-column {
    height: 56vh !important;
    min-height: 0 !important;
    max-height: none !important;
}

#chat-area-wrapper {
    flex: 1 1 0%;
    min-height: 0 !important;
    max-height: 100% !important;
    padding: 6px 10px !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    margin-bottom: 0 !important;
}

#chat-area {
    flex: 1 1 0%;
    max-height: 100%;
    min-height: 0 !important;
}

<input-wrapper> {
    flex-direction: column;
    align-items: stretch;
    gap: 6px;
    padding: 6px 0 0 0 !important;
    border-radius: 0 !important;
    margin-top: 0;
    margin-bottom: 0;
    width: 100% !important;
    box-sizing: border-box !important;
}

.input-wrapper .gr-textbox,
.input-wrapper .gr-button {
    width: 100% !important;
    min-width: 0 !important;
    max-width: 100% !important;
    flex: 1 1 0%;
    box-sizing: border-box !important;
}

.input-wrapper .gr-button {
    min-width: 0 !important;
    max-width: 100% !important;
}    #sidebar {
        height: 30vh !important;
        max-width: 320px !important;
        width: 100% !important;
        border-bottom: 2px solid #eee;
        padding: 18px 16px 16px 16px !important;
        box-sizing: border-box;
        display: flex !important;
        flex-direction: column !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
}
"""

# ------ Additional CSS to set max width and center chat area/input wrapper ------
CHAT_MAXWIDTH_CSS = """
/* ----- Centered chat area with maximum width for desktop ----- */
@media (min-width: 700px) {
  #chat-column {
    align-items: center; /* center children horizontally */
  }
  #chat-area-wrapper {
    max-width: 900px;
    width: 100%;
    margin-left: auto;
    margin-right: auto;
  }
  .input-wrapper {
    max-width: 900px;
    width: 100%;
    margin-left: auto;
    margin-right: auto;
  }
}

/* ----- Center Nexa sidebar content on desktop only ----- */
@media (min-width: 700px) {
  #sidebar, #sidebar * {
    text-align: center !important;
  }
  #sidebar {
    align-items: center !important; /* center children horizontally within sidebar */
    justify-content: flex-start !important; /* preserve top-to-bottom flow */
  }
}

/* ----- Make examples list fill remaining sidebar space on desktop ----- */
@media (min-width: 700px) {
  #examples-list {
    flex: 1 1 auto !important;  /* grow to fill leftover space */
    max-height: none !important; /* remove 300px cap from base */
    min-height: 0 !important;
    overflow-y: auto !important;
  }
}
"""

# ----- Desktop-only fix to ensure examples list fills remaining space and clear button stays at bottom -----
DESKTOP_SIDEBAR_FILL_FIX_CSS = """
@media (min-width: 700px) {
  /* Sidebar as a proper flex column container */
  #sidebar {
    display: flex !important;
    flex-direction: column !important;
    min-height: 0 !important;        /* allow children to shrink */
    overflow: hidden !important;      /* contain children; examples list will scroll */
  }

  /* Make the examples list fill remaining space but be shrinkable and scrollable */
  #examples-list {
    flex: 1 1 0% !important;         /* 0% basis ensures it can shrink to make room */
    min-height: 0 !important;         /* critical so it can shrink below content size */
    max-height: none !important;
    overflow-y: auto !important;      /* scroll the list, not the sidebar */
  }

  /* Keep Clear Chat inside sidebar at the bottom */
  .clear-btn {
    flex: 0 0 auto !important;
    margin-top: auto !important;      /* push it to the bottom within the sidebar */
  }
}
"""

# ----- Center the entire layout (row containing sidebar + chat) on all screen sizes -----
CENTER_ALWAYS_CSS = """
/* Keep overall content centered regardless of viewport size */
#main-row {
  max-width: 1200px;                 /* 280 sidebar + 20 gap + 900 chat content */
  margin-left: auto !important;
  margin-right: auto !important;
  justify-content: center;           /* center the two columns inside the row */
}
"""

# ----- Mobile-only: center chat area and input row with max width 360px -----
MOBILE_CENTERED_360_CSS = """
@media (max-width: 700px) {
  /* Center the chat column children */
  #chat-column {
    align-items: center !important;
  }

  /* Cap width and keep fluid down to small sizes */
  #chat-area-wrapper,
  .input-wrapper {
    max-width: 360px !important;
    width: 100% !important;
    margin-left: auto !important;
    margin-right: auto !important;
  }
}
"""

# ----- Enforce sidebar height to exactly 30% of viewport on mobile -----
MOBILE_SIDEBAR_30VH_ENFORCER_CSS = """
@media (max-width: 700px) {
  /* Enforce 30% of the visible viewport height on mobile */
  #sidebar { 
    height: 30vh !important;   /* legacy fallback */
    height: 30svh !important;  /* small viewport unit fallback */
    height: 30dvh !important;  /* dynamic viewport height (correct on mobile) */
    /* Prevent flex from stretching it beyond 30% */
    flex: 0 0 30dvh !important;

    /* Scroll inside if content is taller */
    overflow-y: auto !important;
  }
}
"""

# ----- Mobile-only: show top heading above sidebar and hide the original heading -----
MOBILE_TOP_HEADING_CSS = """
/* Show the duplicate heading only on mobile */
@media (max-width: 700px) {
  #mobile-top-heading { display: block !important; }
  #chat-column .heading { display: none !important; }
}
/* Hide the duplicate heading on larger screens */
@media (min-width: 701px) {
  #mobile-top-heading { display: none !important; }
}
"""

# ----- Desktop-only: stop examples list from stretching when it has few items -----
# Updated: size to content up to a cap, then scroll; keep Clear Chat pinned.
DESKTOP_EXAMPLES_LIST_NO_STRETCH_CSS = """
@media (min-width: 700px) {
  #examples-list {
    /* Don't stretch to fill; size to content up to a sensible cap, then scroll */
    flex: 0 1 auto !important;
    max-height: clamp(240px, 60vh, 720px) !important;
    min-height: 80px !important;
    overflow-y: auto !important;
  }

  /* Keep Clear Chat pinned at the bottom */
  .clear-btn {
    margin-top: auto !important;
    flex: 0 0 auto !important;
  }
}
"""

# ----- Mobile-only: prevent first heading in examples from tucking under sidebar title -----
# Also keep padding minimal; margin-collapse wasn't the root cause.
MOBILE_SIDEBAR_EXAMPLES_LIST_FIX_CSS = """
@media (max-width: 700px) {
  /* Add small breathing room at the top of the scrollable list */
  #examples-list {
    padding-top: 10px !important;
  }
  /* Neutralize the default top margin of the first child (e.g., h4) */
  #examples-list > *:first-child {
    margin-top: 0 !important;
  }
}
"""

# ----- Mobile-only: explicitly reset flex alignment inside examples list so it starts at the top -----
MOBILE_EXAMPLES_LIST_ALIGNMENT_RESET_CSS = """
@media (max-width: 700px) {
  #examples-list {
    justify-content: flex-start !important;  /* counteracts #sidebar * rule */
    align-items: stretch !important;         /* make children fill width */
  }
  #examples-list > * {
    align-self: stretch !important;          /* ensure each child stretches */
  }
}
"""

# ----- NEW: Ensure headings render and are aligned/visible inside examples list on all screens -----
EXAMPLES_LIST_HEADINGS_VISIBILITY_FIX_CSS = """
/* Make sure headings inside the examples list are visible and not centered/collapsed */
#examples-list {
  justify-content: flex-start !important;
  align-items: stretch !important;
  padding-top: 10px !important; /* keep first heading inside the box */
}

#examples-list > * {
  align-self: stretch !important;
}

/* Remove any unexpected top offset on the very first child */
#examples-list > *:first-child {
  margin-top: 0 !important;
}

/* Override sidebar-wide centering for headings specifically */
#examples-list .gr-markdown,
#examples-list .gr-markdown * {
  text-align: left !important;
}

/* Compact and clear spacing for category headings */
#examples-list .gr-markdown h1,
#examples-list .gr-markdown h2,
#examples-list .gr-markdown h3,
#examples-list .gr-markdown h4,
#examples-list .gr-markdown h5,
#examples-list .gr-markdown h6 {
  margin: 8px 10px 4px 10px !important;
}
"""

# ----- NEW: Unconditional reset to keep headings visible inside examples list -----
EXAMPLES_LIST_COMPONENT_RESET_CSS = """
/* Ensure headings inside the examples list render and aren’t centered/collapsed by #sidebar * rules */
#examples-list {
  display: flex !important;
  flex-direction: column !important;
  align-items: stretch !important;
  justify-content: flex-start !important;
  padding-top: 10px !important; /* keep first heading inside the border */
}
#examples-list > * {
  align-self: stretch !important;
  flex: 0 0 auto !important;     /* don’t let Gradio wrappers auto-grow/shrink oddly */
}

/* Neutralize sidebar-wide centering on markdown wrappers and their inner container */
#examples-list .gr-markdown {
  display: block !important;
  overflow: visible !important;
  margin: 0 !important;
  padding: 0 !important;
  align-items: stretch !important;
  justify-content: flex-start !important;
}
#examples-list .gr-markdown > * {
  display: block !important;     /* inner div that holds the rendered HTML */
  margin: 0 !important;
}

/* Actual heading elements */
#examples-list .gr-markdown h1,
#examples-list .gr-markdown h2,
#examples-list .gr-markdown h3,
#examples-list .gr-markdown h4,
#examples-list .gr-markdown h5,
#examples-list .gr-markdown h6 {
  display: block !important;
  text-align: left !important;
  margin: 8px 10px 4px 10px !important;
  color: inherit !important;
}

/* Keep first visible child from tucking under the top border */
#examples-list > *:first-child {
  margin-top: 0 !important;
}
"""

# ----- NEW: Force user bubble color across Gradio versions (no other changes) -----
CHATBOT_BUBBLE_COLOR_FIX_V4_CSS = """
/* Ensure user/bot message bubble colors apply on Gradio 3.x and 4.x */

/* User bubble (legacy structures) */
#chat-area .message.user,
#chat-area .message.user .bubble,
#chat-area .wrap .message.user,
#chat-area .bubble.user,
/* User bubble (v4 structures with data-testid/role) */
#chat-area [data-testid="user"] .message,
#chat-area [data-testid="user"] .bubble,
#chat-area .message[data-testid="user"],
#chat-area .message[data-role="user"],
#chat-area .chat-message.user,
#chat-area .chat-message.user .message-content,
#chat-area .message.user .message-content {
  background-color: #d4bbf0 !important;
  color: #000 !important;
  border-radius: 16px 16px 0 16px !important;
}

/* Bot bubble (keep consistent with your existing styles) */
#chat-area .message.bot,
#chat-area .message.bot .bubble,
#chat-area .wrap .message.bot,
#chat-area .bubble.bot,
#chat-area [data-testid="bot"] .message,
#chat-area [data-testid="bot"] .bubble,
#chat-area .message[data-testid="bot"],
#chat-area .message[data-role="bot"],
#chat-area .chat-message.bot,
#chat-area .chat-message.bot .message-content,
#chat-area .message.bot .message-content {
  background-color: #f2f2f2 !important;
  color: #000 !important;
  border-radius: 16px 16px 16px 0 !important;
}

/* Optional: set theme variables if Chatbot uses them (harmless if not present) */
#chat-area {
  --chatbot-user-message-background: #d4bbf0 !important;
  --chatbot-user-message-color: #000000 !important;
  --chatbot-bot-message-background: #f2f2f2 !important;
  --chatbot-bot-message-color: #000000 !important;
}
"""

# ----- FINAL: Highest-specificity user bubble override to kill theme’s orange -----
CHATBOT_USER_BUBBLE_FORCE_OVERRIDE_CSS = """
/* Hard override for any Gradio theme that paints user bubbles with accent/orange */
:root, .gradio-container, #chat-area {
  --chatbot-user-message-background: #d4bbf0 !important;
  --chatbot-user-message-color: #000000 !important;
}

/* Cover common v4/v3 DOM variants; remove gradient/background-image if present */
#chat-area .gr-chatbot .message.user .message-content,
#chat-area .gr-chatbot .message.user,
#chat-area .chat-message.user .message-content,
#chat-area .message.user .message-content,
#chat-area .message.user,
.gradio-container #chat-area .message.user .message-content,
.gradio-container #chat-area .message.user,
#chat-area [data-role="user"] .message-content,
#chat-area [data-testid="user"] .message-content,
#chat-area [data-testid="message-user"],
#chat-area [data-testid="message-user"] .message-content {
  background: #d4bbf0 !important;
  background-color: #d4bbf0 !important;
  background-image: none !important;
  color: #000 !important;
  border: none !important;
  border-radius: 16px 16px 0 16px !important;
  box-shadow: none !important;
}

/* Ensure nested elements don’t reintroduce theme backgrounds */
#chat-area .message.user .message-content * {
  background: transparent !important;
}
"""

# ----- NEW: Make user chat bubbles the same purple as the Submit button accent (#7e4f9e) -----
USER_BUBBLE_MATCH_SUBMIT_BTN_CSS = """
/* Make user message bubbles match the Example buttons' light purple */
:root, .gradio-container, #chat-area {
  --chatbot-user-message-background: #d4bbf0 !important;
  --chatbot-user-message-color: #000000 !important;
}

/* Apply across Gradio 3.x/4.x DOM variants with high specificity */
#chat-area .gr-chatbot .message.user .message-content,
#chat-area .gr-chatbot .message.user,
#chat-area .chat-message.user .message-content,
#chat-area .chat-message.user,
#chat-area .message.user .message-content,
#chat-area .message.user,
.gradio-container #chat-area .message.user .message-content,
.gradio-container #chat-area .message.user,
#chat-area [data-role="user"] .message-content,
#chat-area [data-role="user"],
#chat-area [data-testid="user"] .message-content,
#chat-area [data-testid="user"],
#chat-area [data-testid="message-user"] .message-content,
#chat-area [data-testid="message-user"] {
  background: #d4bbf0 !important;
  background-color: #f5f5f5 !important;
  background-image: none !important;
  color: #000000 !important;
  border: none !important;
  border-radius: 16px 16px 0 16px !important;
  box-shadow: none !important;
}

/* Prevent nested elements from reapplying themed backgrounds */
#chat-area .message.user .message-content * {
  background: transparent !important;
  color: inherit !important;
}
"""

with gr.Blocks(css=BASE_CSS
                    + RESPONSIVE_CHAT_CONTAINER_CSS
                    + CHAT_MAXWIDTH_CSS
                    + DESKTOP_SIDEBAR_FILL_FIX_CSS
                    + CENTER_ALWAYS_CSS
                    + MOBILE_CENTERED_360_CSS
                    + MOBILE_SIDEBAR_30VH_ENFORCER_CSS
                    + MOBILE_TOP_HEADING_CSS
                    + DESKTOP_EXAMPLES_LIST_NO_STRETCH_CSS
                    + MOBILE_SIDEBAR_EXAMPLES_LIST_FIX_CSS
                    + MOBILE_EXAMPLES_LIST_ALIGNMENT_RESET_CSS
                    + EXAMPLES_LIST_HEADINGS_VISIBILITY_FIX_CSS
                    + EXAMPLES_LIST_COMPONENT_RESET_CSS
                    + CHATBOT_BUBBLE_COLOR_FIX_V4_CSS
                    + CHATBOT_USER_BUBBLE_FORCE_OVERRIDE_CSS
                    + USER_BUBBLE_MATCH_SUBMIT_BTN_CSS) as demo:

    sidebar_hidden = gr.State(False)

    # Mobile-only duplicate heading above the sidebar (hidden on desktop)
    gr.Markdown("<div class='heading'>Solve with Nexa</div>", elem_id="mobile-top-heading")

    with gr.Row(elem_id="main-row"):
        with gr.Column(elem_id="sidebar"):
            gr.Markdown("### 🤖 Nexa Sidebar")
            gr.Markdown("**Try clicking an example:**")

            with gr.Column(elem_id="examples-list"):
                # Short headings for example categories
                grouped_examples = [
                    ("📐 Functions", [
                        "What’s sin 60?",
                        "cos(π/3 radians)",
                        "inverse sine of 1",
                        "factorial of 5",
                        "What is the exponential of 1?",
                        "What is the base-10 logarithm of 1000?",
                    ]),
                    ("➕ Arithmetic", [
                        "What is five plus three?",
                        "Multiply 6 by 7",
                        "Subtract 10000 by 2000",
                        "Half of 98 plus 12",
                        "Double of 10 and multiply by 3",
                    ]),
                    ("📊 Advanced Math", [
                        "What is the square root of 144?",
                        "Tell me the cube root of 27",
                        "Calculate 3 to the power of 4",
                        "Which is greater: 2^5 or 5^2?",
                        "Is 3² + 4² equal to 5²?",
                    ]),
                    ("📏 Geometry", [
                        "Volume of a sphere with radius 4",
                        "Area of a circle with radius 10",
                        "Area of a triangle with base 10 and height 5",
                        "Volume of a cylinder with radius 3 and height 7",
                    ]),
                    ("🌡 Temperature", [
                        "Convert 100 Celsius to Fahrenheit",
                        "What is 32 Fahrenheit in Celsius?",
                        "Convert 300 Kelvin to Celsius",
                    ]),
                    ("🔄 Conversions", [
                        "Convert 5 feet to meters",
                        "5 miles in kilometers",
                        "200 grams to ounces",
                        "How many inches in 3.5 feet?",
                    ]),
                    ("📅 Date Calculations", [
                        "How many days between Jan 1, 2022 and March 1, 2022?",
                        "What date is 30 days after January 15, 2023?",
                        "What date is 2 weeks before June 5, 2024?",
                    ]),
                ]

                # Flattened lists maintained for event wiring below
                examples = []
                example_buttons = []
                for heading, group in grouped_examples:
                    gr.Markdown(f"#### {heading}")
                    for ex in group:
                        examples.append(ex)
                        example_buttons.append(gr.Button(ex))

            clear_btn = gr.Button("🧹 Clear Chat", elem_classes="clear-btn")

        with gr.Column(elem_id="chat-column"):
            gr.Markdown("<div class='heading'>Solve with Nexa</div>")

            # MOVE .input-wrapper OUTSIDE chat-area-wrapper
            with gr.Group(elem_id="chat-area-wrapper"):
                chatbot = gr.Chatbot(elem_id="chat-area")

            # Input row is now outside so always visible (especially on mobile)
            with gr.Row(elem_classes="input-wrapper"):
                user_input = gr.Textbox(placeholder="Type a math problem...", label="", scale=9)
                submit_btn = gr.Button("Submit", elem_classes="submit-btn", scale=1)  # keep class

            # Events
            submit_btn.click(chat, [user_input, chatbot], [chatbot, user_input])
            user_input.submit(chat, [user_input, chatbot], [chatbot, user_input])

            for btn, ex in zip(example_buttons, examples):
                btn.click(lambda history, _ex=ex: chat(_ex, history),
                          inputs=[chatbot], outputs=[chatbot, user_input])

            clear_btn.click(clear_chat, outputs=[chatbot, user_input])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)

demo.launch()