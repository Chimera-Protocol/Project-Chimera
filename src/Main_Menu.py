# src/Main_Menu.py (GÜNCELLENMİŞ VERSİYON)

import streamlit as st
import os

def main():
    
    st.set_page_config(
        page_title="Project Chimera - Main Menu",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "Welcome to Project Chimera, an ecosystem for strategic AI agents.",
            "Get Help": None,
            "Report a bug": None,
        }
    )

    # --- YENİ EKLENEN CSS KISMI ---
    # Bu CSS bloğu, sütunların içindeki tüm kutuların aynı yüksekliğe sahip olmasını sağlar.
    st.markdown("""
        <style>
            [data-testid="column"] > div > div {
                height: 100%;
            }
        </style>
        """, unsafe_allow_html=True)
    # --- CSS KISMI BİTTİ ---

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    image_path = os.path.join(PROJECT_ROOT, "assets", "main_menu_banner.png")
    if os.path.exists(image_path):
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            st.image(image_path, use_container_width=True)
            

    st.markdown("<h1 style='text-align: center;'>Welcome to Project Chimera</h1>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center; font-weight: normal;'>An Ecosystem for Trustworthy & Strategic AI Agents</h3>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align: center; color: #666; margin-top: -15px;'>v1.7.0 — The Great Unification • Powered by CSL-Core</p>", unsafe_allow_html=True)
    
    st.success(
    "**Welcome!** This is the central hub for Project Chimera's interactive applications. "
    "Please select an experience below.\n\n"
    "🛡️ **System Status:** All agent policies are powered by "
    "[CSL-Core](https://github.com/chimera-protocol/csl-core); "
    "formally verified with Z3 before any action is evaluated. "
    "No rule can contradict another. No unsafe action can slip through."
)

    st.divider()
    col1, col2, col3 = st.columns(3)

    #Adaptive Strategy Lab
    with col1:
        with st.container(border=True):
            st.subheader("🔬 Adaptive Strategy Lab")
            st.markdown(
                "A deep-dive analysis tool for a single agent. "
                "Define a strategic goal, watch the agent think, and explore its decisions with an **interactive XAI dashboard**."
            )
            st.caption("*(Use this for detailed, single-agent analysis.)*")
            st.page_link(
                "pages/1_🔬_Adaptive_Strategy_Lab.py", 
                label="Launch Strategy Lab",
                icon="🔬",
                use_container_width=True
            )

    #The Colosseum
    with col2:
        with st.container(border=True):
            st.subheader("⚔️ The Colosseum")
            st.markdown(
                "An interactive battle arena! Assemble a team of AI gladiators with custom names and doctrines, "
                "and watch them compete against each other in a **live, gamified simulation**."
            )
            st.caption("*(Use this for multi-agent competitive analysis and fun!)*")
            st.page_link(
                "pages/2_⚔️_Colosseum.py",
                label="Enter The Colosseum",
                icon="⚔️",
                use_container_width=True
            )

    # Governance Lab
    with col3:
        with st.container(border=True):
            st.subheader("🏛️ Governance Lab (DEMO)")
            st.markdown(
                "A live demonstration of on-chain governance. Compare competing treasury proposals and "
                "use Chimera to **predict the causal outcome of your vote** before you cast it."
            )
            st.caption("*(The demo built for future steps!)*")
            st.page_link(
                "pages/3_🏛️_Governance_Lab.py",
                label="Enter Governance Lab",
                icon="🏛️",
                use_container_width=True
            )
            
    

if __name__ == "__name__":
    main()