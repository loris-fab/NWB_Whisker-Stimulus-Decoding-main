import gradio as gr
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import src.share as share  # contient MATCHED_FILES_WR_PLUS / WR_MINUS

with gr.Blocks() as multi_tab_app:
    gr.Markdown("# Interface *Oryshchuk et al., 2024, Cell Reports* for WR(-) and WR(+) Trial Template Matching")

    with gr.Tabs():
        # Onglet WR(+)
        if share.MATCHED_FILES_WR_PLUS:  # seulement si des fichiers sont dispo
            with gr.TabItem("Matching WR(+)"):
                from .tabs.temp_matching.interface import app as temp_matching_app
                temp_matching_app.render()
        else:
            with gr.TabItem("Matching WR(+)"):
                gr.Markdown("⚠️ No WR(+) mice selected.")

        # Onglet WR(-)
        if share.MATCHED_FILES_WR_MINUS:
            with gr.TabItem("Matching WR(-)"):
                from .tabs.temp_matching.interface2 import app as temp_matching_app2
                temp_matching_app2.render()
        else:
            with gr.TabItem("Matching WR(-)"):
                gr.Markdown("⚠️ No WR(-) mice selected.")

multi_tab_app.launch(share=True)
