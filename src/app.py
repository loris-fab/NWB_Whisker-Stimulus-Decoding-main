import gradio as gr
from .tabs.temp_matching.interface import app as temp_matching_app
from .tabs.temp_matching.interface2 import app as temp_matching_app2
import warnings
warnings.filterwarnings("ignore", category=UserWarning)




with gr.Blocks() as multi_tab_app:
    gr.Markdown("# Interface *Oryshchuk et al., 2024, Cell Reports* for for WR(-) and WR(+) Trial Template Matching")
    with gr.Tabs():
        with gr.TabItem("Matching WR(+)"):
            temp_matching_app.render()

        with gr.TabItem("Matching WR(-)"):
            temp_matching_app2.render()

multi_tab_app.launch()
