from ..temp_matching.utils import normalize_mat_by_row,get_psth_sum
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import umap
import warnings

# Suppress UMAP warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="'force_all_finite' was renamed to 'ensure_all_finite'")
warnings.filterwarnings("ignore", category=UserWarning, message="using precomputed metric; inverse_transform will be unavailable")
warnings.filterwarnings("ignore", category=UserWarning, message="Transforming new data with precomputed metric. We are assuming the input data is a matrix of distances from the new points to the points in the training set.")

def generate_behavior_corr_plot(trial_results, binary_pred):
    trial_results = pd.Series(trial_results)

    hit_stim = np.sum((trial_results == 'hit') & binary_pred)
    hit_nostim = np.sum((trial_results == 'hit') & (~binary_pred))

    miss_stim = np.sum((trial_results == 'miss') & binary_pred)
    miss_nostim = np.sum((trial_results == 'miss') & (~binary_pred))

    fa_stim = np.sum((trial_results == 'FA') & binary_pred)
    fa_nostim = np.sum((trial_results == 'FA') & (~binary_pred))

    behavior_decoding_corr = np.array([[hit_stim,hit_nostim],
                                    [miss_stim,miss_nostim],
                                    [fa_stim,fa_nostim]])
    fig, ax = plt.subplots()
    sns.heatmap(normalize_mat_by_row(behavior_decoding_corr),square=True,annot=True,cmap='Blues',xticklabels=['Stim','No Stim'],yticklabels=['Hit','Miss','False Alarm'],ax=ax)
    plt.ylabel('Trial Type')
    plt.xlabel('Prediction')
    plt.close()
    return fig

def generate_clustering_plot(trial_templates, trial_stim_amps, distance_metric, centroid_templates, template_stim_amps, incorrect_indices=None):
    """
    Projects trial_templates into 2D using UMAP and plots them, coloring each point based on trial_stim_amp.
    Optionally, centroid_templates can also be plotted with distinct colors and legends.
    Incorrectly classified points can be highlighted if their indices are provided.
    """
    # Perform UMAP dimensionality reduction
    temp_distances = pairwise_distances(X=trial_templates, metric=distance_metric)
    reducer = umap.UMAP(metric='precomputed', n_neighbors=5, min_dist=0, n_jobs=-1)
    trial_templates_2d = reducer.fit_transform(temp_distances)

    # Transform centroid templates into the same 2D space
    if centroid_templates is not None:
        centroid_distances = pairwise_distances(X=centroid_templates, Y=trial_templates, metric=distance_metric)
        centroid_2d = reducer.transform(centroid_distances)

    # Create the plot
    fig = plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        trial_templates_2d[:, 0], trial_templates_2d[:, 1],
        c=trial_stim_amps, cmap="Reds", s=50, edgecolor="black", alpha=0.8
    )

    BORDER_WIDTH = 1
    # Highlight incorrectly classified points
    if incorrect_indices is not None:
        plt.scatter(
            trial_templates_2d[incorrect_indices, 0], trial_templates_2d[incorrect_indices, 1],
            facecolors='none', edgecolors='blue', s=100, label="Incorrectly Classified", linewidth=BORDER_WIDTH
        )

    # Plot centroids with distinct colors and legends
    if centroid_templates is not None:
        centroid_colors = sns.color_palette("Blues", 5)  # Generate distinct colors
        for i, stim_amp in enumerate(template_stim_amps):
            plt.scatter(
                centroid_2d[i, 0], centroid_2d[i, 1],
                c=[centroid_colors[stim_amp]], marker="X", s=200, label=f"Centroid (Stim Amp {stim_amp})", edgecolor="black",linewidths=BORDER_WIDTH
            )

    # Add colorbar for the graded coloring
    cbar = plt.colorbar(scatter)
    cbar.set_label("Stim Amp", fontsize=12)
    cbar.set_ticks(np.unique(trial_stim_amps))  # Set color bar ticks to unique integer values of stim_amps

    # Add plot aesthetics
    plt.title("UMAP Projection of Trial Templates with Centroids", fontsize=16)
    plt.xlabel("UMAP-1", fontsize=12)
    plt.ylabel("UMAP-2", fontsize=12)
    plt.legend(title="Legend", fontsize=10, title_fontsize=12, loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.close()
    return fig  # Return the figure

def plot_confusion_matrix(true_stim_amps, binary_pred):
    cm = np.zeros(shape=(5,2))
    for i in range(cm.shape[0]):
        for j in [0,1]:
            cm[i,j] = ((true_stim_amps == i) & (binary_pred == bool(j))).sum()

    fig, ax = plt.subplots()
    sns.heatmap(normalize_mat_by_row(cm),square=True,cmap='Blues',annot=True,ax=ax)
    plt.xlabel('Stim Prediction')
    plt.ylabel('Stim Amp')
    plt.title('Stimulus Classification Result\nNormalized Per Row')
    ax.set_xticklabels(['No Stim','Stim'])
    ax.set_yticklabels(range(5))
    plt.close()
    return fig

def generate_psth_plot(stim_amps,template_stim_amps,neural_activity,trial_onsets,time_bound,bin_size):
    bin_edges = np.arange(start=time_bound[0],stop=time_bound[1],step=bin_size) 
    fig, axs = plt.subplots(nrows=1, ncols=len(template_stim_amps), figsize=(30, 4),dpi=300,sharey=True)
    for stim_amp_idx,stim_amp in enumerate(template_stim_amps):
        stim_onsets = trial_onsets[stim_amps == stim_amp]
        neuron_psth = get_psth_sum(neural_activity,stim_onsets,bin_size,time_bound,smooth=False)
        peak_index = np.argmax(neuron_psth)
        peak_value = neuron_psth[peak_index]
        peak_time = bin_edges[peak_index]
        ax = axs.flat[stim_amp_idx]
        ax.plot(bin_edges[:-1],neuron_psth,label='PSTH')
        ax.axvline(0,c='r',label='Onset')
        ax.set_xlim(time_bound)
        # mark peak!
        ax.scatter(peak_time,peak_value,marker='^',c='orange',zorder=3,label='Peak')
        ax.set_title(f'stim level: {stim_amp}')
        ax.set_xlabel('time (sec)')
        ax.legend()
    plt.close()
    return fig

def generate_roc_plot(y_true, y_scores,chosen_threshold):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Find the index of the chosen threshold
    threshold_idx = (abs(thresholds - chosen_threshold)).argmin()
    chosen_fpr = fpr[threshold_idx]
    chosen_tpr = tpr[threshold_idx]

    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Random Guess')

    # Mark the chosen threshold point
    plt.scatter([chosen_fpr], [chosen_tpr], color='red', label=f'Threshold = {chosen_threshold:.2f}')
    plt.axvline(x=chosen_fpr, color='red', linestyle='--', lw=1, alpha=0.7)
    plt.axhline(y=chosen_tpr, color='red', linestyle='--', lw=1, alpha=0.7)

    # Annotate the chosen threshold point
    plt.text(chosen_fpr, chosen_tpr, f'({chosen_fpr:.2f}, {chosen_tpr:.2f})', 
             fontsize=10, color='red', ha='right', va='bottom')

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.close()
    return fig

def generate_softscore_hist_plot(trial_scores, trial_stim_amps):
    # softscore for each stim amp in time seq
    # hist of softscore per each stim amp
    fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True, sharey=True)  # Adjust figure size and share axes

    # Use a pleasing color palette
    palette = sns.color_palette("Set2", 5)

    for stim_amp in range(5):
        trial_filt = trial_stim_amps == stim_amp
        y = trial_scores[trial_filt]  # Filtered trial scores

        ax = axes[stim_amp]
        sns.histplot(y, bins=20, kde=True, stat='probability', ax=ax, color=palette[stim_amp], edgecolor="black")  # Plot density

        # Calculate and plot the mean
        mean_value = np.mean(y)
        std_dev = np.std(y)  # Calculate standard deviation (wideness metric)
        ax.axvline(mean_value, color='red', linestyle='--', linewidth=1.5, label=f"Mean: {mean_value:.2f}")
        ax.axvline(mean_value + std_dev, color='blue', linestyle='--', linewidth=1.5, label=f"Std Dev: {std_dev:.2f}")
        ax.axvline(mean_value - std_dev, color='blue', linestyle='--', linewidth=1.5)  # Plot negative std deviation

        # Force the y-axis limits to update after plotting
        ax.set_ylim(0, ax.get_ylim()[1])  # Explicitly set the lower limit to 0

        # Get updated y-axis limits for the current subplot
        y_min, y_max = ax.get_ylim()

        # Add yellow tint between the two STD lines
        ax.fill_betweenx(
            y=[y_min, y_max],  # Fill from the bottom to the top of the y-axis
            x1=mean_value - std_dev,
            x2=mean_value + std_dev,
            color='yellow',
            alpha=0.2  # Adjust transparency
        )

        # Add legends for mean and standard deviation
        ax.legend(loc="upper right")  # Add the mean and std dev as separate legends

        ax.set_title(f'Stim Amp: {stim_amp}')
        ax.set_xlim([0, 1])  # Assuming trial scores are normalized between 0 and 1

    # Add shared x-axis and y-axis labels
    fig.text(0.55, 0.04, 'Trial Scores', ha='center', fontsize=12)  # Adjusted y position for better centering
    fig.text(0.04, 0.5, 'Density', va='center', rotation='vertical', fontsize=12)  # Updated label to 'Density'

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.close()
    return fig  # Return the figure

def generate_softscore_overtime_plot(trial_scores, trial_stim_amps):
    # softscore for each stim amp in time seq
    # hist of softscore per each stim amp
    fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True, sharey=True)  # Adjust figure size and share axes
    slopes = []

    palette = sns.color_palette("Set2", 5)

    for stim_amp in range(5):
        trial_filt = trial_stim_amps == stim_amp
        x = np.arange(len(trial_scores[trial_filt]))
        y = trial_scores[trial_filt]

        ax = axes[stim_amp]
        sns.regplot(x=x, y=y, ax=ax, scatter_kws={'s': 10, 'color': palette[0]}, 
                    line_kws={'color': 'red', 'linestyle': '--'})
        slope = np.polyfit(x, y, 1)[0]  # Extract slope from regression
        slopes.append(slope)
        ax.legend([f"Slope: {slope:.2f}"], loc="upper right")
        ax.set_title(f'Stim Amp: {stim_amp}')

        ax.set_ylim([0, 1])

    fig.text(0.55, 0.04, 'Trial Occurrence', ha='center', fontsize=12)  # Adjusted y position for better centering
    fig.text(0.04, 0.5, 'Perception', va='center', rotation='vertical', fontsize=12)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.close()
    return fig

def generate_temp_distance_plot(stim_templates, sim_metric):
    fig, ax = plt.subplots()
    temp_distances = pairwise_distances(X=stim_templates,metric=sim_metric)
    sns.heatmap(temp_distances/temp_distances.max(),cmap='Blues',square=True,ax=ax)
    plt.title('Normalized Pairwise Distance of Templates')
    plt.xlabel('Stim Level')
    plt.ylabel('Stim Level')
    plt.close()
    return fig


def plot_custom_psth(
    data_struct,
    session_idx,
    align_event,                 # "lick_timestamps" | "jaw_dlc_licks" | "trial_onset"
    use_lick_index,              # CheckboxGroup -> list de [True/False] (peut être [])
    stim_amp,                    # CheckboxGroup -> liste d'amps sélectionnées
    result,                      # CheckboxGroup -> liste de response types sélectionnés
    rpv_thresh, isi_thresh, iso_thresh,
    brain_region, ccf_acronym, neuron_type,  # CheckboxGroup (listes)
    time_bound,                  # (ms_start, ms_end)
    bin_size                     # (ms)
):
    """
    Computes and returns a Matplotlib figure of a custom PSTH.
    - time_bound and bin_size are in ms (converted to seconds internally).
    - All neuron filters are applied.
    - Alignment options: trial_onset / jaw_dlc_licks / lick_timestamps
      * lick_timestamps: aligns on transitions (onsets or offsets) of lick_indices.
    """

    s = int(session_idx)

    # --- Raccourcis vers la session ---
    trials = data_struct['trials'][s]

    # --- Récupère les tableaux unités (mêmes longueurs) ---
    # Convertis en np.array pour masques logiques
    rpv = np.asarray(trials.get('fractionRPVs_estimatedTauR', []), dtype=float)
    isi = np.asarray(trials.get('isi_violation', []), dtype=float)
    iso = np.asarray(trials.get('iso_distance', []), dtype=float)

    br  = np.asarray(trials.get('brain_region', []), dtype=object)
    ccf = np.asarray(trials.get('ccf_names_acronyms', []), dtype=object)
    typ = np.asarray(trials.get('Type_of_neuron', []), dtype=object)

    spikes_list = np.asarray(trials.get('spikes', []), dtype=object)
    n_units = len(spikes_list)

    if n_units == 0:
        fig = plt.figure()
        plt.text(0.5, 0.5, "Aucun neurone dans cette session.", ha='center', va='center')
        plt.axis('off')
        plt.close()
        return fig

    # --- Masque neurones (RPV ≤, ISI ≤, ISO ≥, appartenance aux groupes cochés) ---
    unit_mask = np.ones(n_units, dtype=bool)

    if rpv.size == n_units:
        unit_mask &= (rpv <= float(rpv_thresh))
    if isi.size == n_units:
        unit_mask &= (isi <= float(isi_thresh))
    if iso.size == n_units:
        unit_mask &= (iso >= float(iso_thresh))

    if isinstance(brain_region, list) and len(brain_region) > 0 and br.size == n_units:
        unit_mask &= np.isin(br, brain_region)

    if isinstance(ccf_acronym, list) and len(ccf_acronym) > 0 and ccf.size == n_units:
        unit_mask &= np.isin(ccf, ccf_acronym)

    if isinstance(neuron_type, list) and len(neuron_type) > 0 and typ.size == n_units:
        unit_mask &= np.isin(typ, neuron_type)

    # --- Neurones retenus ---
    kept_spikes = spikes_list[unit_mask]
    if len(kept_spikes) == 0:
        fig = plt.figure()
        plt.text(0.5, 0.5, "Filtres trop stricts : 0 neurone retenu.", ha='center', va='center')
        plt.axis('off')
        plt.close()
        return fig

    # --- Sélection des évènements d’alignement ---
    # Conversion ms -> s
    TB = (float(time_bound[0]) / 1000.0, float(time_bound[1]) / 1000.0)
    BIN = float(bin_size) / 1000.0

    def _safe_get_top_level(key):
        # data_struct peut être "fusionné" (listes par session) ou session unique
        if key in data_struct:
            val = data_struct[key]
            # si c'est une liste par session
            if isinstance(val, (list, tuple)) and len(val) > s:
                return val[s]
            return val
        # fallback: parfois stocké dans trials (peu probable)
        return trials.get(key, None)

    events = None

    if align_event == "trial_onset":
        trial_onsets = _safe_get_top_level('trial_onset')
        if trial_onsets is None:
            fig = plt.figure()
            plt.text(0.5, 0.5, "Aucun 'trial_onset' disponible.", ha='center', va='center')
            plt.axis('off')
            plt.close()
            return fig

        stim_amp_arr = np.asarray(trials.get('stim_amp', []))
        result_arr   = np.asarray(trials.get('result', []), dtype=object)

        if stim_amp_arr.size == 0 or result_arr.size == 0:
            fig = plt.figure()
            plt.text(0.5, 0.5, "Métadonnées de trials manquantes (stim_amp/result).", ha='center', va='center')
            plt.axis('off')
            plt.close()
            return fig

        mask = np.ones_like(stim_amp_arr, dtype=bool)
        if isinstance(stim_amp, list) and len(stim_amp) > 0:
            mask &= np.isin(stim_amp_arr, stim_amp)
        if isinstance(result, list) and len(result) > 0:
            mask &= np.isin(result_arr, result)

        events = np.asarray(trial_onsets)[mask]

    elif align_event == "jaw_dlc_licks":
        jaw_evt = _safe_get_top_level('jaw_dlc_licks')
        if jaw_evt is None:
            fig = plt.figure()
            plt.text(0.5, 0.5, "Aucun 'jaw_dlc_licks' disponible.", ha='center', va='center')
            plt.axis('off')
            plt.close()
            return fig
        events = np.asarray(jaw_evt)

    elif align_event == "lick_timestamps":
        # On aligne sur les transitions de lick_indices:
        #  - True => transitions 0->1 (début léchage)
        #  - False => transitions 1->0 (fin léchage)
        t = _safe_get_top_level('lick_timestamps')
        li = np.asarray(trials.get('lick_indices', []), dtype=bool)

        if t is None or li.size == 0 or len(t) != len(li):
            fig = plt.figure()
            plt.text(0.5, 0.5, "LickTrace indisponible ou dimensions incohérentes.", ha='center', va='center')
            plt.axis('off')
            plt.close()
            return fig

        li_i = li.astype(np.int8)
        prev = np.concatenate(([0], li_i[:-1]))
        on_edges  = (li_i == 1) & (prev == 0)  # 0->1
        off_edges = (li_i == 0) & (prev == 1)  # 1->0
        on_times  = np.asarray(t)[on_edges]
        off_times = np.asarray(t)[off_edges]

        # use_lick_index est une liste venant d’un CheckboxGroup
        def _has_true(x):  return isinstance(x, list) and (True in x)
        def _has_false(x): return isinstance(x, list) and (False in x)

        if _has_true(use_lick_index) and _has_false(use_lick_index):
            events = np.sort(np.concatenate([on_times, off_times]))
        elif _has_true(use_lick_index):
            events = on_times
        elif _has_false(use_lick_index):
            events = off_times
        else:
            # Si rien sélectionné, par défaut on prend les onsets
            events = on_times
    else:
        fig = plt.figure()
        plt.text(0.5, 0.5, f"Alignement inconnu: {align_event}", ha='center', va='center')
        plt.axis('off')
        plt.close()
        return fig

    # --- Vérifs évènements ---
    if events is None or len(events) == 0:
        fig = plt.figure()
        plt.text(0.5, 0.5, "Aucun évènement sélectionné avec ces filtres.", ha='center', va='center')
        plt.axis('off')
        plt.close()
        return fig

    # --- Calcul PSTH (moyenne population) ---
    # Reprend la logique de get_psth_sum utilisée ailleurs (si tu l'as importée, tu peux l’appeler directement)
    bin_edges = np.arange(start=TB[0], stop=TB[1], step=BIN)
    hist_sum = np.zeros(len(bin_edges) - 1, dtype=float)

    # Boucle neurones -> histogramme aligné -> normalisation par nb trials et bin
    n_trials = len(events)
    for neuron_spikes in kept_spikes:
        neuron_spikes = np.asarray(neuron_spikes, dtype=float)
        # Accumule l’histo sur tous les évènements
        h = np.zeros(len(bin_edges) - 1, dtype=float)
        for onset in events:
            rel = neuron_spikes - onset
            mask = (rel >= TB[0]) & (rel <= TB[1])
            h += np.histogram(rel[mask], bins=bin_edges)[0]
        # taux/s (diviser par nb évènements * largeur de bin)
        if n_trials > 0:
            h = h / (n_trials * BIN)
        hist_sum += h

    # Moyenne sur neurones
    hist_mean = hist_sum / len(kept_spikes)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot(bin_edges[:-1], hist_mean, label='PSTH')
    ax.axvline(0.0, linestyle='--', linewidth=1, label='Align', alpha=0.7)
    ax.set_xlim(TB)
    ax.set_xlabel('Temps (s)')
    ax.set_ylabel('Taux de décharge (Hz)')
    title_parts = [f"Session {s}", f"Align: {align_event}", f"Units: {len(kept_spikes)}", f"Events: {len(events)}"]
    ax.set_title(" | ".join(title_parts))
    ax.legend(loc='best')
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.close()
    return fig
