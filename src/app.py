import gradio as gr
from .tabs.temp_matching.interface import app as temp_matching_app

with gr.Blocks() as multi_tab_app:
    with gr.Tab("Template Construction"):
        temp_matching_app.render() 
multi_tab_app.launch()