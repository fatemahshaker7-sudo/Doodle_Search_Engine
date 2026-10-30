# 🎨 DoodleSearch – AI-Powered Doodle Search Engine

**Live Demo:** [DoodleSearch – Streamlit App](https://doodlesearch.streamlit.app/)  

---

## Problem Statement

Finding visually similar sketches is not as straightforward as matching photographs. Doodles are often simple, abstract, and drawn differently by different people, even when they represent the same object.

Traditional image-matching methods that compare individual pixels can struggle with variations in drawing style, shape, and positioning.

DoodleDex addresses this challenge by using a trained deep-learning model to represent doodles as numerical feature embeddings and retrieve visually similar drawings from a gallery. Users can either draw a doodle directly in the application or upload an image to discover similar results.

The application focuses on visual similarity rather than displaying category labels.

---

## Executive Summary

DoodleDex is an image-retrieval application that uses deep-learning embeddings to find doodles that resemble a user's drawing.

The project follows an image-retrieval pipeline:

1. **Data Preparation:** Load and organize the doodle dataset into gallery, query, and held-out sets.
2. **Preprocessing:** Convert input drawings into a consistent format before passing them to the model.
3. **Feature Extraction:** Use a trained deep-learning model to transform each doodle into a numerical embedding.
4. **Index Construction:** Generate and save gallery embeddings alongside their corresponding images.
5. **Similarity Search:** Compare a query embedding with gallery embeddings and retrieve the most similar doodles.
6. **Evaluation:** Measure retrieval performance using Precision@K and evaluate a similarity threshold to distinguish familiar queries from held-out examples.
7. **Deployment:** Build an interactive Streamlit application that supports both drawing and image uploads.

The application uses precomputed gallery embeddings to avoid rebuilding the search index every time a user opens the app.

Rather than returning category predictions, DoodleSearch presents a visual gallery of the most similar doodles, making the retrieval experience intuitive and interactive.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Executive Summary](#executive-summary)
- [Project Structure](#project-structure)
- [Dataset and Data Preparation](#dataset-and-data-preparation)
- [Methodology](#methodology)
- [Evaluation](#evaluation)
- [Streamlit Application](#streamlit-application)
- [Key Visualizations](#key-visualizations)
- [Conclusions and Recommendations](#conclusions-and-recommendations)
- [Areas for Further Research](#areas-for-further-research)
- [Technologies Used](#technologies-used)

---

## Project Structure

```text
Doodle_Search_Engine/
│
├── app/
│   ├── app.py
│   ├── search_core.py
│   └── requirements.txt
│
├── index/
│   ├── embeddings.npy
│   ├── gallery_images.npy
│   ├── gallery_labels.npy
│   └── config.json
│
├── quickdraw-search/
│   ├── prepare_data.py
│   ├── build_index.py
│   └── evaluate.py
│
├── images/
│   ├── good_bad_results.png
│   ├── grid_check.png
│   ├── threshold_hist.png
│   └── preprocess_check.png
│
└── README.md
```

*Note: The structure above illustrates the separation between the deployed application, saved search index, and model/data preparation scripts. Adjust filenames and folders to match the actual repository.*

### Main files

| File | Description |
|---|---|
| `app.py` | Streamlit application for drawing, uploading, and retrieving similar doodles |
| `search_core.py` | Contains the model-loading, preprocessing, embedding, and similarity-search functions |
| `requirements.txt` | Python dependencies required to run the application |
| `embeddings.npy` | Precomputed feature embeddings for the gallery |
| `gallery_images.npy` | Gallery images corresponding to the saved embeddings |
| `gallery_labels.npy` | Gallery labels corresponding to the saved embeddings |
| `config.json` | Search configuration, including the similarity threshold and number of results |
| `prepare_data.py` | Prepares the dataset for the retrieval pipeline |
| `build_index.py` | Generates and saves the gallery embeddings |
| `evaluate.py` | Evaluates retrieval performance and determines the search threshold |

---

## Dataset and Data Preparation

The project organizes doodle images into three main groups:

| Dataset Split | Purpose |
|---|---|
| Gallery | Reference images used for similarity search |
| Queries | Images used to evaluate retrieval performance |
| Held-out | Separate examples used to evaluate whether the system can distinguish unfamiliar inputs |

### Preprocessing

Before an image is passed to the model, it must be transformed into the expected input format.

The preprocessing pipeline ensures consistency between gallery images and user-submitted drawings.

The application supports:
- Resizing drawings to the model's expected input dimensions.
- Converting images into a consistent pixel representation.
- Handling foreground and background colors.
- Preparing user-uploaded images and canvas drawings using the same preprocessing pipeline.

---

## Methodology

### 1. Feature Extraction

A trained deep-learning model transforms each doodle into a numerical feature vector called an **embedding**.

Instead of comparing images directly pixel by pixel, the system compares these learned representations to identify visually similar drawings.

### 2. Gallery Indexing

The model generates embeddings for the gallery images.

These embeddings are saved in `embeddings.npy`, while the corresponding images are saved in `gallery_images.npy`.

This allows the application to retrieve results without recalculating all gallery embeddings during every session.

### 3. Similarity Search

When a user draws or uploads an image:

1. The image is preprocessed.
2. The trained model generates its embedding.
3. The query embedding is compared with the saved gallery embeddings.
4. The 25 most similar gallery images are retrieved.
5. The application displays the top results.

The search process uses similarity scores to rank the retrieved doodles.

### 4. Similarity Threshold

A similarity threshold is used to determine whether the closest gallery match is sufficiently similar to the query.

If the highest similarity score falls below the configured threshold, the application displays a message indicating that no sufficiently similar doodle was found.

This helps prevent the application from presenting weak matches as reliable results.

---

## Evaluation

### Precision@K

The main retrieval evaluation metric is **Precision@K**.

It measures the proportion of the top K retrieved images that belong to the same category as the query image.

\[
Precision@K =
\frac{\text{Relevant images in top K results}}{K}
\]

For example, if 18 out of the top 25 retrieved images belong to the query's category:

\[
Precision@25 = \frac{18}{25} = 0.72
\]

This means that 72% of the retrieved images are relevant according to their category.

Precision@K is calculated for individual queries and then averaged to evaluate overall retrieval performance.

### Similarity Threshold Evaluation

The threshold is evaluated using query and held-out similarity scores.

The objective is to balance:
- Accepting queries that have suitable matches in the gallery.
- Rejecting held-out examples that do not have sufficiently reliable matches.

The selected threshold is saved in `config.json` and used by the deployed application.

---

## Streamlit Application

DoodleSearch is deployed using Streamlit and provides two ways to interact with the search engine.

### Draw

Users can create their own doodles directly on an interactive drawing canvas.

The application preprocesses the drawing and searches for similar images in the gallery.

### Upload

Users can upload an existing PNG or JPG image instead of drawing.

The uploaded image goes through the same preprocessing and retrieval pipeline.

### Results

The application displays:
- The user's original input.
- A preview of the preprocessed image.
- The most similar doodles retrieved from the gallery.
- Similarity scores for the retrieved results.

The application focuses on visual retrieval and does not display category labels.

**Try the application:** [DoodleSearch Live Demo](https://doodlesearch.streamlit.app/)

---

## Key Visualizations

Suggested screenshots to include:

1. **Drawing Interface:** The interactive canvas where users create doodles.
2. **Image Upload:** The interface for uploading an existing drawing.
3. **Retrieval Results:** An example showing the input doodle and its most similar gallery images.
4. **Evaluation Results:** Precision@K comparisons and similarity-score distributions used during model evaluation.

---

## Conclusions and Recommendations

### Conclusions

- Deep-learning embeddings provide a feature-based representation for comparing doodles.
- Precomputing gallery embeddings makes the retrieval process more efficient during application use.
- Precision@K provides a way to measure how relevant the retrieved images are to each query.
- A similarity threshold helps control which results are presented as sufficiently similar.
- Streamlit provides an accessible interface for testing image retrieval through both drawing and uploading.

### Recommendations

- Continue evaluating retrieval performance across different doodle categories and drawing styles.
- Test how preprocessing choices affect retrieval quality.
- Explore approximate nearest-neighbor search to support larger galleries.
---

## Areas for Further Research

- **Improved Embeddings:** Experiment with alternative CNN architectures and metric-learning approaches to improve visual similarity.
- **Data Augmentation:** Test rotation and scaling, to improve robustness to different drawing styles.
- **Threshold Calibration:** Investigate category-specific thresholds and more diverse held-out evaluation data.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Deep-learning model and embedding generation |
| NumPy | Numerical operations and saved embedding arrays |
| Pillow (PIL) | Image loading and processing |
| Streamlit | Interactive web application |
| Streamlit Drawable Canvas | Drawing interface |
| Scikit-learn | Evaluation metrics, where used |

---

## Sources & Acknowledgements

- **Dataset:** [Google Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset) – Doodle images provided under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
- **Pretrained Model:** [MobileNetV2 – Keras Applications](https://keras.io/api/applications/mobilenet/#mobilenetv2-function) – Used as a feature extractor to generate image embeddings.
- **Drawing Interface:** [Streamlit Drawable Canvas](https://github.com/andfanilo/streamlit-drawable-canvas) – Interactive doodle drawing functionality.
- **Web Application:** [Streamlit Documentation](https://docs.streamlit.io/) – Application development and deployment.
- **Project Reference:** Quick, Draw! Doodle Search Engine activity.
