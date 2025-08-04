

# Template Matching GUI – Adapted for NWB Files

**Study:** Oryshchuk et al., 2024, *Cell Reports*

This project is an extension of an EPFL semester project focused on decoding perception of whisker stimuli in mice during an operant conditioning task.

I adapted the original project to support **NWB-formatted data** from the publication *Oryshchuk et al., 2024*, and extended the GUI for more detailed and interactive **PSTH (Peri-Stimulus Time Histogram)** visualization.

## 📚 Reference

Oryshchuk et al., *Distributed and specific encoding of sensory, motor, and decision information in the mouse neocortex during goal-directed behavior*, Cell Reports, 2024.  
👉 [DOI](https://doi.org/10.1016/j.celrep.2023.113618)


## ⚙️ Features

* **Support for NWB files**
* **Interactive GUI via Gradio for Dual analysis tabs: WR(+) and WR(–) trials**
* PSTH visualization aligned to trial onset, jaw movement... / Custom filters: stimulus amplitudes, neuron types, brain regions, quality metrics (RPV, ISI, ISO)
* Decoder parameter tuning and behavioral alignment
* Continuous perception estimation over time


## 🚀 Usage

Create environment and Install dependencies with:
```bash
conda env create -f environment.yml
```

if it doesn't work try :
```bash
conda create -n ao-visu-310 python=3.10
conda activate ao-visu-310
pip install gradio gradio_rangeslider scipy scikit-learn pymatreader gdown pynwb matplotlib seaborn umap-learn
```



Download NWB files from the **LSENS Laboratory of Sensory Processing**, and put them inside the folder:


```
./NWB_files
```

## 🧩 How to use

```bash
python -m src.app
```

Then open the local URL provided in your browser.

## ✍️ Author

The original codebase was created by **Sobhan Nili**, and has been adapted for this project by @loris-fab.

For questions related to this adaptation, you can contact: loris.fabbro@epfl.ch

---

