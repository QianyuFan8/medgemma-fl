# Action Items (19-Oct)

1.  Launch the GPU environment and run the COMET ERMS/ARMS experiments --\> BLOCKED
2.  Spaced H&E tile selection for the MedGemma montage --\> COMPLETED
3.  Confirm the training-only limma CpG panel (top 0.1%) --\> COMPLETED
4.  Split the result summary into prediction, compute, and communication --\> COMPLETED
5.  Add a MyoD-mutant class --\> WAITING ON MUTATION LABELS

# Updates (19-Oct-2026)

-   **Cohort and labels**
    -   Patient-tumor UMAP, top 5% variable CpGs: the two large groups line up with histology. 232/246 ERMS sit in the left group and 143/158 ARMS sit in the right group.
    -   Among samples with a fusion call, all 108 fusion-positive cases are ARMS and none are ERMS. Eight ARMS are fusion-negative. 161 ERMS and 42 ARMS have no fusion result, so fusion status is not a substitute label.
    -   The small top group is not “all Spindle + RMS NOS.” It mixes Spindle, RMS NOS, ERMS, and Unknown. The tables have no MYOD1 mutation field, so that group is not being relabeled MyoD mutant.
-   **H&E montage**
    -   Tile embedding is unchanged: tissue detection, 224 px at 0.5 μm/px, frozen H-Optimus-1, gated ABMIL trained on training slides only.
    -   The 16 tiles sent to MedGemma are no longer the 16 highest-attention tiles in a row. Tiles are taken from high to low attention, and each new tile must be at least two tile-widths from those already chosen. If that does not fill 16, the remaining slots use the next-highest tiles.
    -   Image scripts now live in `data_preprocessing/`.
-   **Methylation panel**
    -   Detection p \> 0.01 is set to missing, CpGs missing in more than 5% of training samples are dropped, and the remaining gaps are filled with the training-set median.
    -   Limma is fit on training M-values only. The panel is the top 0.1% by p-value, about 847 CpGs. This is not being expanded to 1% or 5%.
-   **Experiments that will be run**
    -   01 H-Optimus-1 + ABMIL, image only.
    -   02 MedGemma 1.5 zero-shot, no training.
    -   03 MedGemma 1.5 QLoRA, centralized, one client, rank 16.
    -   04 MedGemma 1.5 QLoRA, federated, three clients, rank 16: IID, then skew 0.8.
    -   Evaluation is on the validation split first. The summary prints one table for ROC-AUC, PR-AUC, accuracy, balanced accuracy, and class recall, and a second table for train time, GPU hours, peak GPU memory per site, rounds, upload, download, and total traffic.
-   **Code and compute**
    -   The current pipeline is on branch `comet-erms-arms`. `COMET_RMS_Data/` is gitignored.
    -   GPU launch is blocked. NVIDIA Brev reports a service disruption for creating, starting, and accessing environments. The planned machine is 4× L40S 48 GB, 512 GB disk, stoppable, named `medgemma-fl`. Nothing is running, so there is no GPU charge.
