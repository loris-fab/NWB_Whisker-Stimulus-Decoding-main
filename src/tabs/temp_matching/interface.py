
#%%
# Default interface values
DEFAULT_SESSION = 0
DEFAULT_BIN_SIZE = 10
DEFAULT_CONF_THRESH = 0.5
DEFAULT_TIMEBOUND = (0,100)
DEFAULT_ESTIM_BIN_SIZE = 100
PARALLEL_RUN = True
#%%
import gradio as gr
from gradio_rangeslider import RangeSlider
from ..temp_matching.callbacks import *
from .utils import *

data_struct = load_data()

######## CE QUE J'AI RAJOUTER ########
with open("data_struct_nwb.pkl", "rb") as f:
    data_struct = pickle.load(f)
#######################################

with gr.Blocks() as app:
    gr.Markdown("## Template Matching Analysis")

    gr.Markdown("### ● Session Selection")
    max_session = len(data_struct['date']) - 1
    session_idx = gr.Number(label="Session Index", value=DEFAULT_SESSION, precision=0, interactive=True, minimum=0, maximum=max_session)
    session_label_display = gr.Textbox(label="Session Name", interactive=False,value=f"{data_struct['mouse'][DEFAULT_SESSION]} - {data_struct['date'][DEFAULT_SESSION].strftime('%Y-%m-%d')}")
    session_neurons_count = gr.Textbox(label="Neurons Count", interactive=False,value=update_neurons_count(data_struct,DEFAULT_SESSION))
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Article: *Oryshchuk et al., 2024, Cell Reports*")
        with gr.Column():
            gr.File(value="src/static/2024_Oryshchuk_CellReports.pdf", label="Browse Article", file_types=[".pdf"], interactive=False)
    gr.Markdown("#### Response Types")
    with gr.Row():
        with gr.Column():
            session_hit_count = gr.Textbox(label="Hit Trials", interactive=False,value=update_trial_type_count(data_struct,DEFAULT_SESSION,'hit'))
            session_miss_count = gr.Textbox(label="Miss Trials", interactive=False,value=update_trial_type_count(data_struct,DEFAULT_SESSION,'miss'))
        with gr.Column():
            session_CR_count = gr.Textbox(label="CR Trials", interactive=False,value=update_trial_type_count(data_struct,DEFAULT_SESSION,'CR'))
            session_FA_count = gr.Textbox(label="FA Trials", interactive=False,value=update_trial_type_count(data_struct,DEFAULT_SESSION,'FA'))

    gr.Markdown("### ● Analysis Parameters")
    gr.Markdown("#### General Parameters")
    time_bound = RangeSlider(label="Time Bound (ms)", minimum=0, maximum=500, step=1, value=DEFAULT_TIMEBOUND)
    bin_size = gr.Slider(label="Bin Size (ms)", minimum=5, maximum=200, step=10, value=DEFAULT_BIN_SIZE)

        # --- PSTH Custom Interface ---
    gr.Markdown("#### Custom PSTH parameters")

    with gr.Row():
        psth_align_event = gr.Dropdown(label="Align PSTH on", choices=["lick_timestamps", "trial_onset", "reaction_time"], value="trial_onset", interactive=True)
        if psth_align_event.value == "trial_onset":
            psth_use_stim_filter = gr.Checkbox(label="Apply Stim Filter (only trials with stim)", value=False, interactive=True)

    with gr.Row():
        psth_brain_region = gr.Dropdown(label="Brain Region", choices=sorted(list(set(data_struct['trials'][DEFAULT_SESSION]['brain_region']))), interactive=True)
        psth_stim_amp = gr.Dropdown(label="Stim Amplitude", choices=np.unique(data_struct['trials'][DEFAULT_SESSION]['stim_amp']).tolist(), interactive=True)
        psth_result = gr.Dropdown(label="Response Type", choices=sorted(set(data_struct['trials'][DEFAULT_SESSION]['result'])), value="hit", interactive=True)

    gr.Markdown("#### Other analysis parameters")
    with gr.Row():
        with gr.Column():
            confidence_threshold = gr.Slider(label="Confidence Threshold", minimum=0, maximum=1, step=0.01, value=DEFAULT_CONF_THRESH,interactive=False)
            optimize_threshold = gr.Checkbox(label='Optimize threshold for F1',value=True)
    with gr.Row():
        with gr.Column():
            similarity_metric = gr.Dropdown(label="Similarity Metric", choices=["cosine", "euclidean"], value="cosine")
            feature_type = gr.Dropdown(label="Feature Type", choices=[ft.name for ft in FeatureType], value=[ft.name for ft in FeatureType][0])
            multi_level = gr.Checkbox(label="Enable Multi-Level Templates", value=False)
        with gr.Column():
            stim_temp_trial_type = gr.Dropdown(label="Stim Trial Types", choices=["All", "Hit", "Miss"], value="All")
            nostim_temp_trial_type = gr.Dropdown(label="No Stim Trial Types", choices=["All", "CR", "FA"], value="All")
            show_wrong_labels = gr.Checkbox(label='Indicate wrong classifications',value=True)

    with gr.Row():
        save_results_indicator = gr.Checkbox(label="Calculate Continuous Predictions", value=False)
        estim_bin_size = gr.Slider(label="Estimation Bin Size (ms)", minimum=10, maximum=500, step=100, value=DEFAULT_ESTIM_BIN_SIZE)
        estim_timepoints_count = gr.Textbox(label="Timepoints Count", interactive=False,value=int(data_struct['trial_onset'][DEFAULT_SESSION].max()//estim_bin_size.value*1000) if data_struct is not None else '')
    



    """
    psth_run_button = gr.Button("Plot Custom PSTH")
    #psth_custom_plot = gr.Plot(label="Custom PSTH")
    psth_run_button.click(fn=plot_custom_psth,inputs=[session_idx, psth_align_event, psth_use_stim_filter, psth_brain_region, psth_stim_amp, psth_result],outputs=psth_output_plot)
    """
    
    run_button = gr.Button("Run Analysis")
    temp_plot_output = gr.Plot(label="PSTH of Stim-Amps")

    with gr.Row():
        temp_distance_plot_output = gr.Plot(label="Template Distances",visible=False)
        temp_clustering_plot_output = gr.Plot(label="Template Representations")
        roc_plot_output = gr.Plot(label="ROC Curve")
        confmat_plot_output = gr.Plot(label="Confusion Matrix")
        behavior_corr_plot_output = gr.Plot(label="Behaviour Correspondance")
    with gr.Row():
        perception_overtime_plot_output = gr.Plot(label="Perception of Stim over Time")
        perception_hist_plot_output = gr.Plot(label="Perception Distribution")
    f1_score_output = gr.Textbox(label="F1 Score", interactive=False)
    download_output = gr.File(label="Save Continuous Perception",visible=False)

    # Events
    optimize_threshold.change(toggle_confidence_slider, inputs=[optimize_threshold], outputs=[confidence_threshold])
    save_results_indicator.change(lambda save: gr.update(visible=save),inputs=[save_results_indicator],outputs=[download_output])
    session_idx.change(lambda inputs: update_neurons_count(data_struct,inputs), inputs=[session_idx], outputs=[session_neurons_count])
    session_idx.change(lambda session_idx: update_trial_type_count(data_struct,session_idx,'hit'), inputs=[session_idx], outputs=[session_hit_count])
    session_idx.change(lambda session_idx: update_trial_type_count(data_struct,session_idx,'miss'), inputs=[session_idx], outputs=[session_miss_count])
    session_idx.change(lambda session_idx: update_trial_type_count(data_struct,session_idx,'CR'), inputs=[session_idx], outputs=[session_CR_count])
    session_idx.change(lambda session_idx: update_trial_type_count(data_struct,session_idx,'FA'), inputs=[session_idx], outputs=[session_FA_count])
    session_idx.change(lambda session_idx:gr.update(value=int(data_struct['trial_onset'][session_idx].max()//estim_bin_size.value*1000)),inputs=[session_idx],outputs=[estim_timepoints_count])
    estim_bin_size.change(lambda session_idx,estim_bin_size:gr.update(value=int(data_struct['trial_onset'][session_idx].max()//estim_bin_size*1000)),inputs=[session_idx,estim_bin_size],outputs=[estim_timepoints_count])
    session_idx.change(lambda s: f"{data_struct['mouse'][s]} - {data_struct['date'][s].strftime('%Y-%m-%d')}",inputs=[session_idx],outputs=[session_label_display])
    run_button.click(
        lambda session_idx, similarity_metric, time_bound, multi_level,
            stim_temp_trial_type, nostim_temp_trial_type, bin_size,
            confidence_threshold, optimize_threshold, show_wrong_labels,
            feature_type, save_results_indicator, estim_bin_size: 
            run_analysis(
                data_struct, session_idx, similarity_metric, time_bound, multi_level,
                stim_temp_trial_type, nostim_temp_trial_type, bin_size,
                confidence_threshold, optimize_threshold, show_wrong_labels,
                feature_type, save_results_indicator, estim_bin_size
            ),
        inputs=[
            session_idx, similarity_metric, time_bound, multi_level,
            stim_temp_trial_type, nostim_temp_trial_type, bin_size,
            confidence_threshold, optimize_threshold, show_wrong_labels,
            feature_type, save_results_indicator, estim_bin_size
        ],
        outputs=[
            temp_plot_output, temp_distance_plot_output, temp_clustering_plot_output,
            roc_plot_output, confmat_plot_output, behavior_corr_plot_output,
            perception_overtime_plot_output, perception_hist_plot_output,
            f1_score_output, download_output
        ]
    )
if __name__ == "__main__":
    app.launch()