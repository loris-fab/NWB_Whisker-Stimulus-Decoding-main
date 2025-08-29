
#%%
# Default interface values
DEFAULT_SESSION = 0
DEFAULT_BIN_SIZE = 10
DEFAULT_CONF_THRESH = 0.5
DEFAULT_TIMEBOUND = (-50,200)
DEFAULT_ESTIM_BIN_SIZE = 100
PARALLEL_RUN = True
#%%
import gradio as gr
from gradio_rangeslider import RangeSlider
from ..temp_matching.callbacks import *
from .utils import *
from ... import share



#######################################
# Load data
data_struct = load_all_sessions_merged_Selected_files(SELECTED_FILES=share.MATCHED_FILES_WR_PLUS , Rewarded_choice=True)
#######################################

def S(x):
    try:
        return int(x)
    except Exception:
        return DEFAULT_SESSION


def update_quality_sliders(session_idx):
    def _safe_min_max(arr, default=(0.0, 1.0)):
        if arr is None: return default
        arr = np.asarray(arr)
        if arr.size == 0 or not np.isfinite(arr).any():
            return default
        return float(np.nanmin(arr)), float(np.nanmax(arr))
    s = S(session_idx)
    trials = data_struct['trials'][s]
    rpv_min, rpv_max = _safe_min_max(trials.get('fractionRPVs_estimatedTauR', []), (0, 1))
    isi_min, isi_max = _safe_min_max(trials.get('isi_violation', []), (0, 1))
    iso_min, iso_max = _safe_min_max(trials.get('iso_distance', []), (0, 50))
    return (
        gr.update(minimum=rpv_min, maximum=rpv_max, value=rpv_max),
        gr.update(minimum=isi_min, maximum=isi_max, value=isi_max),
        gr.update(minimum=iso_min, maximum=iso_max, value=iso_min),
    )

def update_alignment_filters_rewarded(event):
    if event == "lick_timestamps":
        return gr.update(visible=True), gr.update(visible=False , value =[])
    elif event == "trial_onset":
        return (
            gr.update(visible=False),
            gr.update(visible=True),
        )
    else:
        return gr.update(visible=False , value =[]), gr.update(visible=False, value =[])

def update_stim_choice(stim_choice, session_idx):
    """
    Update the visibility of the stim_amp_filter based on the selected stim_choice.
    If 'Stimulation' is selected, show the stim_amp_filter; otherwise, hide it.
    """
    s = S(session_idx)
    trials = data_struct['trials'][s]

    if stim_choice == "Stimulation":
        amps = sorted(set(a for a in map(float, trials['stim_amp']) if a > 0))
        result = sorted(set(r for r in trials['result'] if r not in ("CR", "FA")))
        return gr.update(visible=True, choices=[str(a) for a in amps], value=[]), gr.update(visible=True, choices=result, value=[])
    elif stim_choice == "No-Stimulation":
        amps = sorted(set(a for a in map(float, trials['stim_amp']) if a <= 0))
        result = sorted(set(r for r in trials['result'] if r in ("CR", "FA")))
        return gr.update(visible=True, choices=amps, value=amps), gr.update(visible=True, choices=result, value=[])
    else:
        return gr.update(visible=False, value=[]), gr.update(visible=False , value=[])


def update_target_area_filter(target_area_filter, session_idx):
    """
    Update the visibility of the ccf_acronym_filter and type_of_neuron_filter based on the selected target_area_filter.
    If 'All' is selected, show both filters; otherwise, hide them.
    """
    s = S(session_idx)
    trials = data_struct['trials'][s]
    list_brain_region = np.asarray(trials['brain_region'], dtype=str)
    indices = [i for i, region in enumerate(list_brain_region) if region in target_area_filter]

    ccf_acronyms = sorted(set(np.asarray(trials['ccf_names_acronyms'])[indices]))

    return gr.update(visible=True, choices=ccf_acronyms, value=[])


def update_ccf_acronym_filter(ccf_acronym_filter, session_idx):
    """
    Update the visibility of the type_of_neuron_filter based on the selected ccf_acronym_filter.
    If 'All' is selected, show the type_of_neuron_filter; otherwise, hide it.
    """
    s = S(session_idx)
    trials = data_struct['trials'][s]
    list_ccf_acronyms = np.asarray(trials['ccf_names_acronyms'], dtype=str)
    indices = [i for i, acronym in enumerate(list_ccf_acronyms) if acronym in ccf_acronym_filter]

    neuron_types = sorted(set(np.asarray(trials['Type_of_neuron'])[indices]))

    return gr.update(visible=True, choices=neuron_types, value=[])


def update_session_filters_rewarded(session_idx):
    s = S(session_idx)
    trials = data_struct['trials'][s]
    amps    = sorted(set(map(float, trials['stim_amp'])))
    results = sorted(set(trials['result']))
    neuron_types = sorted(set(trials['Type_of_neuron']))
    regions      = sorted(set(trials['brain_region']))
    ccf_acros    = sorted(set(trials['ccf_names_acronyms']))
    return (
        gr.update(choices=[str(a) for a in amps], value=[]),
        gr.update(choices=results, value=[]),
        gr.update(choices=neuron_types, value=[]),
        gr.update(choices=regions, value=[]),
        gr.update(choices=ccf_acros, value=[]),
    )


with gr.Blocks() as app:

    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Article: *Oryshchuk et al., 2024, Cell Reports*")
        with gr.Column():
            gr.File(value="src/static/2024_Oryshchuk_CellReports.pdf", label="Browse Article", file_types=[".pdf"], interactive=False)

    gr.Markdown("### ● Session Selection")
    max_session = len(data_struct['date']) - 1
    session_idx = gr.Number(label="Session Index", value=DEFAULT_SESSION, precision=0, interactive=True, minimum=0, maximum=max_session)
    session_label_display = gr.Textbox(label="Session Name", interactive=False,value=f"{data_struct['mouse'][DEFAULT_SESSION]} - {data_struct['date'][DEFAULT_SESSION].strftime('%Y-%m-%d')}")
    session_neurons_count = gr.Textbox(label="Neurons Count", interactive=False,value=update_neurons_count(data_struct,DEFAULT_SESSION))


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
    time_bound = RangeSlider(label="Time Bound (ms)", minimum=-200, maximum=500, step=1, value=DEFAULT_TIMEBOUND)
    bin_size = gr.Slider(label="Bin Size (ms)", minimum=5, maximum=200, step=10, value=DEFAULT_BIN_SIZE)
    
    ######################
    #  psth Interface    #
    ######################
    gr.Markdown("#### ▪ Custom PSTH parameters")
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Choose the alignment event for the PSTH")
            psth_align_event = gr.Dropdown(
                label="Align PSTH on",
                choices=["lick_timestamps", "jaw_dlc_licks", "trial_onset"],
                value="jaw_dlc_licks",
                interactive=True
            )

            lick_filter = gr.CheckboxGroup(label="Lick Indices",choices=[True, False],value=[True], visible=False)



            stim_choice = gr.Dropdown(
                label="Stimulation or No-Stimulation",
                choices=["Stimulation", "No-Stimulation"],
                value= None,
                visible=False,
                interactive=True
            )

            stim_amp_filter = gr.CheckboxGroup(
                label="Stim Amplitudes",
                choices=[],
                value=[],
                visible=False
            )
            
            response_type_filter = gr.CheckboxGroup(
                label="Response Types",
                choices=[],
                value=[],
                visible=False
            )


        with gr.Column():
            gr.Markdown("#### Choose the filters for the PSTH")
            min_rpv = np.asarray(data_struct['trials'][DEFAULT_SESSION]['fractionRPVs_estimatedTauR']).min()
            max_rpv = np.asarray(data_struct['trials'][DEFAULT_SESSION]['fractionRPVs_estimatedTauR']).max()

            min_isi = np.asarray(data_struct['trials'][DEFAULT_SESSION]['isi_violation']).min()
            max_isi = np.asarray(data_struct['trials'][DEFAULT_SESSION]['isi_violation']).max()

            min_iso = np.asarray(data_struct['trials'][DEFAULT_SESSION]['iso_distance']).min()
            max_iso = np.asarray(data_struct['trials'][DEFAULT_SESSION]['iso_distance']).max()

            rpv_step = max((max_rpv - min_rpv)/50, 1e-6)
            isi_step = max((max_isi - min_isi)/50, 1e-6)

            rpv_slider = gr.Slider(minimum=min_rpv, maximum=max_rpv, value=max_rpv, step= rpv_step, label="RPV (%)")
            isi_slider = gr.Slider(minimum=min_isi, maximum=max_isi, value=max_isi, step= isi_step, label="ISI Violation Rate")
            iso_slider = gr.Slider(minimum=min_iso, maximum=max_iso, value=min_iso, step=1, label="Isolation Distance")


            target_area_filter = gr.CheckboxGroup(label="Brain Region", choices=sorted(list(set(data_struct['trials'][DEFAULT_SESSION]['brain_region']))), interactive=True , value= [],visible=True)
            ccf_acronym_filter = gr.CheckboxGroup(label="CCF Acronym", choices=sorted(set(data_struct['trials'][DEFAULT_SESSION]['ccf_names_acronyms'])), interactive=True, value= [],visible = False)
            type_of_neuron_filter = gr.CheckboxGroup(label="Type of Neuron", choices=sorted(set(data_struct['trials'][DEFAULT_SESSION]['Type_of_neuron'])), interactive=True , value= [],visible = False)



    with gr.Row():
        choice_saved = gr.Dropdown(
            label="Saved plot?",
            choices=["Yes", "No"],
            value="No",
            interactive=True
        )


    psth_run_button = gr.Button("Plot Custom PSTH")
    psth_output_plot = gr.Plot(label="Custom PSTH")

    def plot_custom_psth_handler(session_idx, align_event, use_lick_index, stim_amp, result,
                                rpv_thresh, isi_thresh, iso_thresh,
                                brain_region, ccf_acronym, neuron_type,
                                time_bound, bin_size, choice_saved):

        stim_amp_vals = None if not stim_amp else [float(x) for x in stim_amp]

        return plot_custom_psth(
            data_struct, session_idx, align_event, use_lick_index,
            stim_amp_vals, result, rpv_thresh, isi_thresh, iso_thresh,
            brain_region, ccf_acronym, neuron_type, time_bound, bin_size, choice_saved
        )
    
    psth_run_button.click(
        fn=plot_custom_psth_handler,
        inputs=[
            session_idx, psth_align_event, lick_filter,
            stim_amp_filter, response_type_filter,
            rpv_slider, isi_slider, iso_slider,
            target_area_filter, ccf_acronym_filter, type_of_neuron_filter,
            time_bound, bin_size,choice_saved
        ],
        outputs=[psth_output_plot]
    )

    ######################
    #   Other analysis   #
    ######################

    gr.Markdown("#### ▪ Other analysis parameters")
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
    


    run_button = gr.Button("Run Analysis")
    temp_plot_output = gr.Plot(label="PSTH of Stim-Amps", visible=False)

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

    ######################
    #       EVENT        #
    ######################
    # MY EVENTS------------

    session_idx.change(
    update_session_filters_rewarded,
    inputs=[session_idx],
    outputs=[
        stim_amp_filter,
        response_type_filter,
        type_of_neuron_filter,
        target_area_filter,
        ccf_acronym_filter
    ]
    )
    session_idx.change(
    update_quality_sliders,
    inputs=[session_idx],
    outputs=[rpv_slider, isi_slider, iso_slider]
    )

    psth_align_event.change(
    update_alignment_filters_rewarded,
    inputs=[psth_align_event],
    outputs=[lick_filter, stim_choice]
    )
    stim_choice.change(
    update_stim_choice,
    inputs=[stim_choice, session_idx],
    outputs=[stim_amp_filter, response_type_filter]
)

    target_area_filter.change(
        update_target_area_filter,
        inputs=[target_area_filter, session_idx],
        outputs=[ccf_acronym_filter]
    )
    ccf_acronym_filter.change(
        update_ccf_acronym_filter,
        inputs=[ccf_acronym_filter, session_idx],
        outputs=[type_of_neuron_filter]
    )
    # ------------------------
    # OTHER EVENTS
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