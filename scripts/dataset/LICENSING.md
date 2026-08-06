# Base-dataset licensing & redistribution status

Neither base dataset ships a LICENSE file, so **before publishing any merged dataset
that re-hosts their images**, we need the authors' permission. Until then, the safe
release is: OUR OWN photos + annotations + conversion scripts + pointers to the originals
(the current plan, decision D5 in RESEARCH_PLAN.md).

| Dataset | Author / contact | Cite | Status |
|---|---|---|---|
| AngelinaDataset | Ilya Ovodov — via GitHub issue on IlyaOvodov/AngelinaDataset (no public email in repo) | Ovodov, "Optical Braille Recognition Using Object Detection Neural Network", ICCVW 2021 (arXiv:2012.12412) | ☐ not yet asked |
| DSBI | Renqiang Li / Hong Liu — lirenqiang@ict.ac.cn, hliu@ict.ac.cn (ICT, Chinese Academy of Sciences) | Li et al., "DSBI: Double-Sided Braille Image Dataset...", ICVISP 2018 (arXiv:1811.10893) | ☐ not yet asked |

## Draft message (adapt per author)

> Subject: Permission to redistribute [AngelinaDataset / DSBI] annotations in a merged braille detection dataset
>
> Dear [Dr. Ovodov / Dr. Li],
>
> I am a student researcher building an open braille-detection dataset and model
> (fine-tuned RF-DETR, 63-class six-dot encoding). I have converted your
> [AngelinaDataset / DSBI] annotations to COCO format for training, citing
> [ICCVW 2021, arXiv:2012.12412 / ICVISP 2018, arXiv:1811.10893].
>
> I would like to ask whether you would permit redistribution of the images and/or
> converted annotations as part of a merged public dataset (CC-BY-4.0, with full
> attribution and citation), published via Zenodo/GitHub/Hugging Face. If you prefer,
> I will instead publish only conversion scripts that point to your original repository.
>
> Either way, your dataset will be cited in the accompanying paper.
>
> Thank you for making this data available.
> [name, affiliation]

Log replies here. If no reply after ~3 weeks, default to scripts-and-pointers (D5).
