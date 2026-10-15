---
title: 'My paper Detecting Humanized AI Text in Academic Writing Through Mathematical Source Relationships got presented at Ohio Dominican University'
date: 2026-05-15
permalink: /blog/detecting-humanized-ai-text/
tags:
  - presentation
---

---
layout: post
title: "Can AI Help Us Assess Building Damage After a Disaster?"
date: 2026-10-15
categories: [research, computer-vision, machine-learning]
tags: [remote-sensing, disaster-response, xBD, ResNet, Vision-Transformer]
---

*An undergraduate research project exploring computer vision for post-disaster building damage assessment.*

## Why This Problem Matters

After a major disaster, response teams need to understand which buildings may have been damaged and where help is most urgently needed. Satellite imagery can provide a broad view of affected areas, but manually reviewing large amounts of imagery can be time-consuming.

In this project, our team explored whether deep-learning models could classify building damage from satellite imagery. The goal was not to replace trained inspectors, but to investigate how AI might help prioritize areas for further review.

The project was completed as part of the SAIRI 2026 research program. Our team included Alameen Adeku, Asheley Mudzingwa, Christian Messado, and Keren Gil, with graduate mentor John Tanimola and faculty mentor Dr. Steve Efe.

## Research Question

**How do different image-classification architectures perform when classifying post-disaster building damage, especially when damage classes are imbalanced?**

We worked with the xBD post-disaster building damage dataset, which provides pre- and post-disaster satellite imagery and damage labels. The four target classes were:

| Class | Description |
|---|---|
| **No Damage** | No visible building damage |
| **Minor Damage** | Limited visible damage |
| **Major Damage** | Substantial visible damage |
| **Destroyed** | Building appears destroyed |

The xBD dataset was introduced by Gupta et al. (2019). See the [xBD research paper](https://arxiv.org/abs/1911.09296).

## Approach

We compared three deep-learning approaches. The baseline model used a single image, while the Siamese approach was designed to use a pre-disaster and post-disaster image pair. We also evaluated a Vision Transformer (ViT).

![Project pipeline showing the image-to-prediction workflow]({{ '/assets/images/pipeline.png' | relative_url }})

*Figure 1. Overview of the project pipeline, adapted from our final presentation.*

### Models at a Glance

| Model | Input | Core Idea | Training Status |
|---|---|---|---|
| **ResNet-50 baseline** | Single image | ImageNet-pretrained convolutional neural network used as a baseline classifier | 20 epochs |
| **Siamese ResNet-50** | Pre- and post-disaster image pair | Shared feature extractor compares information from both images | 4 epochs; training was ongoing |
| **ViT-B/16** | Image divided into 16 × 16 patches | Transformer architecture models relationships between image patches | 20 epochs |

These models represent different ways of extracting visual information. A single-image model learns patterns from one image, while a paired-image model can potentially use changes between pre- and post-disaster imagery. A transformer provides another approach by modeling relationships between image patches.

### Training Configuration

| Setting | ResNet-50 Baseline | Siamese ResNet-50 | ViT-B/16 |
|---|---|---|---|
| Pretraining | ImageNet | ResNet-50 feature extractor | ImageNet |
| Loss function | Weighted cross-entropy | Weighted cross-entropy | Weighted cross-entropy |
| Optimizer | AdamW | AdamW | AdamW |
| Learning rate listed in configuration | `1e-4`* | `1e-4` | `1e-4` |
| Batch size | 24 | 8 | 12 |
| Epochs reported | 20 | 4 (ongoing) | 20 |

\*The presentation contains an inconsistency: the baseline configuration table lists a learning rate of `1e-4`, while the speaker notes list `1e-5`. This should be checked against the training code before the value is treated as definitive.

We used weighted cross-entropy to address class imbalance during training. Model performance was evaluated using overall accuracy, precision, recall, macro F1, weighted F1, and per-class performance.

These metrics offer different perspectives. Accuracy summarizes the proportion of correct predictions, macro F1 gives each class equal weight, and weighted F1 accounts for class frequency in the evaluation set. Considering several metrics is especially important when some damage classes are less common than others.

## Results

The metrics below are reproduced from our final presentation. They should be interpreted together rather than treating any single metric as a complete description of model quality.

### Overall Model Comparison

| Model | Epochs | Overall Accuracy | Precision | Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 baseline | 20 | 0.8868 | 0.9891 | 0.8868 | 0.2449 | 0.9346 |
| Siamese ResNet-50 | 4 (ongoing) | 0.6000 | 0.7497 | 0.6000 | 0.4621 | 0.6502 |
| ViT-B/16 | 20 | 0.6111 | 0.9889 | 0.6111 | 0.2007 | 0.7540 |

**An important metric check:** the reported baseline accuracy, macro F1, and weighted F1 tell quite different stories. In particular, a high weighted F1 can coexist with poor performance on underrepresented classes. The reported values should be rechecked against the evaluation script and class supports before being used as formal benchmark results.

The Siamese model had completed only four epochs and was still being trained, so its reported metrics should not be interpreted as a final, directly comparable result against models trained for 20 epochs.

### Understanding the Confusion Matrices

Confusion matrices help reveal which damage categories a model recognizes and which categories it confuses with one another.

![ResNet-50 baseline confusion matrix]({{ '/assets/images/baseline-confusion-matrix.png' | relative_url }})

*Figure 2. ResNet-50 baseline confusion matrix from the final presentation.*

The presentation reports that the baseline correctly classified approximately 89% of **No Damage** examples and 81% of **Destroyed** examples. However, it correctly classified only 6% of **Minor Damage** examples and 0% of **Major Damage** examples.

This suggests that the model struggled to distinguish damage severity despite its strong overall accuracy. One possible explanation is that the model favored the majority class, making it less reliable for recognizing less frequent damage categories.

![ViT-B/16 confusion matrix]({{ '/assets/images/vit-confusion-matrix.png' | relative_url }})

*Figure 3. ViT-B/16 confusion matrix from the final presentation.*

The ViT-B/16 results showed a similar challenge. The presentation reports that 61% of **No Damage** examples were correctly classified, while 55% of **Minor Damage** examples and 33% of **Major Damage** examples were predicted as **No Damage**.

This pattern matters in a disaster-response setting. If a model misclassifies a damaged building as undamaged, that prediction could lead to an affected location receiving less attention than it needs. For that reason, evaluating per-class recall and confusion patterns is important in addition to overall accuracy.

### Training Behavior

![Training loss curves]({{ '/assets/images/training-loss.png' | relative_url }})

*Figure 4. Training-loss curves included in the project presentation.*

![Validation weighted F1 curves]({{ '/assets/images/validation-weighted-f1.png' | relative_url }})

*Figure 5. Validation weighted-F1 curves included in the project presentation.*

The training curves provide additional context for interpreting the summary metrics. In future experiments, I would compare checkpoints using classwise precision and recall, macro F1, and confusion matrices—not only weighted F1—so that improvements on the majority class do not hide poor recognition of less frequent damage categories.

## What I Learned

The main takeaway from this project is that **overall accuracy is not enough for imbalanced, high-impact classification tasks**. A model can appear strong on a summary metric while missing the classes that matter most for decision-making.

The experiments also showed me why model comparisons need to be carefully controlled. Because the Siamese model was still being trained, its results cannot establish whether it performs better or worse than the other architectures after equivalent training.

Rather than identifying one universally best architecture, the project highlighted the importance of evaluation methodology, class-specific performance, and transparent reporting.

## Limitations and Future Work

There are several limitations and opportunities for further research.

| Limitation | Potential Next Step |
|---|---|
| Imbalanced damage classes | Investigate focal loss, resampling, and carefully designed data augmentation |
| Incomplete Siamese training | Complete training and evaluate it under a comparable experimental setup |
| Unverified metric and configuration details | Recheck evaluation outputs and training configuration against the code |
| Uncertain generalization to unseen disasters | Evaluate on held-out disaster events, regions, and imaging conditions |
| Image-level classification provides limited spatial detail | Explore segmentation approaches for more localized damage information |
| Predictions are not a substitute for professional inspection | Design the system as a decision-support tool with human review |

A key direction I would like to explore is cross-disaster generalization: whether a model trained on imagery from certain events can reliably assess damage from a different disaster or geographic region. This would help evaluate whether the model learns useful damage patterns or relies too heavily on characteristics of the training data.

Another direction is segmentation, which could provide more spatially specific information about damaged areas than a single classification label.

The project presentation also described a prototype FastAPI deployment intended to return building-safety-oriented classifications. This should be understood as a research prototype, not a certified ATC-20 inspection system or a replacement for trained inspectors.

## Conclusion

This project gave me an opportunity to explore how computer vision might support disaster-response workflows and to see firsthand how dataset imbalance affects model behavior.

The experiments did not establish a universally best architecture. Instead, they highlighted the need for careful evaluation, transparent reporting, and strong per-class performance when model outputs could influence real-world priorities.

As an undergraduate interested in research, I found this project valuable because it connected model development with a practical problem and showed me that building a useful AI system involves more than achieving a high accuracy score. It also requires understanding failure cases, questioning evaluation metrics, and considering how predictions might be used in practice.

I am interested in continuing to study reliable computer vision systems, particularly approaches that combine remote sensing, robust evaluation, and human-centered decision support.

## Acknowledgements

Thanks to my teammates Alameen Adeku, Asheley Mudzingwa, Christian Messado, and Keren Gil; our graduate mentor John Tanimola; and our faculty mentor Dr. Steve Efe for their guidance and collaboration during SAIRI 2026.

## References

1. Gupta, R. et al. (2019). *xBD: A Dataset for Assessing Building Damage from Satellite Imagery.* [arXiv:1911.09296](https://arxiv.org/abs/1911.09296).
2. United Nations Office for Disaster Risk Reduction (2020). *The Human Cost of Disasters: An Overview of the Last 20 Years (2000–2019).* [UNDRR report](https://www.undrr.org/publication/human-cost-disasters-overview-last-20-years-2000-2019).

---

*This post summarizes a student research project and its presentation results. The reported metrics are included as presented and should be independently verified before being used for operational or scientific claims.*


------
