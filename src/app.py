import gradio as gr
import warnings
import importlib
import src.share as share  # for MATCHED_FILES_WR_PLUS / WR_MINUS

warnings.filterwarnings("ignore", category=UserWarning)
importlib.reload(share) 

temp_matching_app = None
temp_matching_app2 = None

if share.MATCHED_FILES_WR_PLUS:
    from .tabs.temp_matching.interface import app as temp_matching_app

if share.MATCHED_FILES_WR_MINUS:
    from .tabs.temp_matching.interface2 import app as temp_matching_app2

# ---- Construire l'interface
with gr.Blocks() as multi_tab_app:
    gr.Markdown("# Interface *Oryshchuk et al., 2024, Cell Reports* for WR(-) and WR(+) Trial Template Matching")

    with gr.Tabs():
        with gr.TabItem("Matching WR(+)"):
            if temp_matching_app:
                temp_matching_app.render()
            else:
                gr.Markdown("⚠️ No WR(+) mice selected.")

        with gr.TabItem("Matching WR(-)"):
            if temp_matching_app2:
                temp_matching_app2.render()
            else:
                gr.Markdown("⚠️ No WR(-) mice selected.")

multi_tab_app.launch(share=True)
