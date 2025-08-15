
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

data_struct = load_all_sessions_merged(share.Folder, Rewarded_choice=False)

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

def update_alignment_filters(event,session_idx):
    s = S(session_idx)
    trials = data_struct['trials'][s]
    if event == "PiezoLickOnsets":
        return gr.update(visible=False , value =[])
    elif event == "Coils_onset":
        amps = sorted(set(a for a in map(float, trials['stim_amp']) if a > 0))
        return gr.update(visible=True, choices=[str(a) for a in amps], value=[])
    else:
        raise ValueError(f"Unknown alignment event: {event}")

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

def update_session_filters(session_idx):
    s = S(session_idx)
    trials = data_struct['trials'][s]
    amps    = sorted(set(map(float, trials['stim_amp'])))
    neuron_types = sorted(set(trials['Type_of_neuron']))
    regions      = sorted(set(trials['brain_region']))
    ccf_acros    = sorted(set(trials['ccf_names_acronyms']))
    return (
        gr.update(choices=[str(a) for a in amps], value=[]),
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

    gr.Markdown("### ● Custom PSTH parameters")
    time_bound = RangeSlider(label="Time Bound (ms)", minimum=-200, maximum=500, step=1, value=DEFAULT_TIMEBOUND)
    bin_size = gr.Slider(label="Bin Size (ms)", minimum=5, maximum=200, step=10, value=DEFAULT_BIN_SIZE)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Choose the alignment event for the PSTH")
            psth_align_event = gr.Dropdown(
                label="Align PSTH on",
                choices=["PiezoLickOnsets","Coils_onset"],
                value = "PiezoLickOnsets",
                interactive=True
            )


            stim_amp_filter = gr.CheckboxGroup(
                label="Stim Amplitudes",
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

    # PSTH Custom Plot
    psth_run_button = gr.Button("Plot Custom PSTH")
    psth_output_plot = gr.Plot(label="Custom PSTH")

    def plot_custom_psth_handler(session_idx, align_event, stim_amp,
                                rpv_thresh, isi_thresh, iso_thresh,
                                brain_region, ccf_acronym, neuron_type,
                                time_bound, bin_size, choice_saved):

        stim_amp_vals = None if not stim_amp else [float(x) for x in stim_amp]

        return plot_custom_psth_non_rewarded(
            data_struct, session_idx, align_event,
            stim_amp_vals, rpv_thresh, isi_thresh, iso_thresh,
            brain_region, ccf_acronym, neuron_type, time_bound, bin_size, choice_saved
        )
    
    psth_run_button.click(
        fn=plot_custom_psth_handler,
        inputs=[
            session_idx, psth_align_event,
            stim_amp_filter,
            rpv_slider, isi_slider, iso_slider,
            target_area_filter, ccf_acronym_filter, type_of_neuron_filter,
            time_bound, bin_size, choice_saved
        ],
        outputs=[psth_output_plot]
    )

    # Events
    session_idx.change(
    update_session_filters,
    inputs=[session_idx],
    outputs=[
        stim_amp_filter,
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
    update_alignment_filters,
    inputs=[psth_align_event,session_idx],
    outputs=[stim_amp_filter]
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

    session_idx.change(lambda inputs: update_neurons_count(data_struct,inputs), inputs=[session_idx], outputs=[session_neurons_count])
    session_idx.change(lambda s: f"{data_struct['mouse'][s]} - {data_struct['date'][s].strftime('%Y-%m-%d')}",inputs=[session_idx],outputs=[session_label_display])



if __name__ == "__main__":
    app.launch()