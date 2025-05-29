# From Totoro to Tangled: analyzing lexical and thematic patterns in Studio Ghibli vs Disney"

## Abstract

This study leverages computational text analysis to explore linguistic and emotional distinctions in the dialogue of 21 Disney films and 17 Studio Ghibli classics. By applying Latent Dirichlet Allocation (LDA) for topic modeling, transformer-based sentiment analysis, and zero-shot thematic classification to subtitle data, we identify recurring themes, emotional trajectories, and gendered language patterns. Our results show that Studio Ghibli’s narratives foreground themes of grief and friendship alongside more balanced gender representation, whereas Disney’s films tend to emphasize loneliness and identity through male-coded language and conflict-driven topics. These findings uncover measurable storytelling differences between the two studios and illustrate the power of NLP methods to reveal subtle but meaningful patterns in cinematic dialogue.  

### Data

In the **data/** folder we can find the following subsections:

- **initial_dataset/**: Contains raw subtitle files for 38 films (21 Disney, 17 Ghibli).

- **data_cleaned/**: Outputs from cleaning, where HTML tags, timestamps, and speaker labels have been removed (run via Data_Preprocessing.ipynb).

- **data_preprocessed/**: Word tokens formatted for LDA.

- **data_in_sentences/**: One sentence per line for sentiment models.

- **data_split/**: Partitioned data for LDA use.


### Notebooks

This repository offers three core notebooks that drive the analysis pipeline from raw subtitles to final insights.

1. **Data_Preprocessing.ipynb**

This notebook serves as your one-stop guide to transforming raw subtitle files into formats ready for analysis. You begin by cleaning each *.srt* file, stripping out HTML tags, timestamps, and speaker labels. Once the text is clean, the notebook demonstrates how to split everything into individual sentences for sentiment analysis and, in parallel, tokenize the full documents for topic modeling. Each intermediate output is automatically saved into its respective *data/* subfolder, so you can inspect or reuse any stage of the pipeline.

2. **LDA_Topic_Modelling.ipynb**

Here you’ll work through topic modeling for both Disney and Studio Ghibli dialogues in parallel. The notebook begins by loading the tokenized corpora and building document-term matrices, then applies LDA separately to the Disney and Ghibli datasets. For each studio, you will see preprocessing of tokens, model fitting, examination of topic-term distributions, and visualizations of top topics. A concluding section compares coherence scores and thematic clusters across the two studios, helping to surface narrative distinctions between Disney and Ghibli.

3. **Sentiment_Analysis.ipynb** 

This notebook explores two sentiment analysis approaches. It starts by importing sentence-level data from *data_in_sentences/*, then applies a transformer-based RoBERTa model to infer sentiment scores and visualize comparative trajectories. Next, it runs a zero-shot thematic classifier to label both emotion and overarching themes in the dialogue. Various configurations and thresholding strategies are demonstrated, with resulting CSV outputs and plots organized under *Sentiment_Analysis/RoBERTa/* and *Sentiment_Analysis/zero-shot classification results/*.

### Results

- **Sentiment_Analysis/RoBERTa/**: CSV files of RoBERTa-based sentiment scores, sentiment comparison plot and progression plots by film.

- **Sentiment_Analysis/zero-shot classification results/**: Theme and sentiment label outputs for individual films.

- **Presentation_Ghibli_vs_Disney.pdf**: Slides summarizing methods, findings, and visualizations.

### Presentation and Report

- **Presentation_Ghibli_vs_Disney.pdf**: Slides summarizing methods, findings, and visualizations.

- **Final_Report.pdf**: comprehensive project report detailing methodology, results, and discussion.


*Navigate the repository, run the notebooks in order, and explore the contrasts between Disney and Studio Ghibli!*