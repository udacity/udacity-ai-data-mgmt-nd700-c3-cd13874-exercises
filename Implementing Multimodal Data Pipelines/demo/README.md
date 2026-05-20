# Multimodal Data Pipeline Demo
### Data Management for Generative AI

This demo shows a **multimodal data pipeline** example used in **Data Management for Generative AI** applications. The goal is to demonstrate how a system can ingest, preprocess, align, and store multiple modalities (text and images) so they can be used in downstream GenAI applications such as Retrieval-Augmented Generation (RAG).

The example processes a **PDF apparel catalog**, extracting text and images, generating image descriptions using a vision-capable LLM, and storing the resulting multimodal records in a vector database for search and question answering.



The implementation is intentionally simplified so students can focus on **data management decisions in multimodal systems**, not only on model usage.

---

# Use Case

Many apparel retailers maintain large digital catalogs that contain dozens or hundreds of pages with images, descriptions, and styling ideas. While these catalogs are visually rich, they are often difficult to search efficiently. A customer, merchandiser, or sales associate might want to quickly answer questions such as “Which outfits are good for a casual outdoor event?”, “Do any pages show baseball-inspired jackets?”, or “Where can I find outfits with red and white color combinations?”. Traditional keyword search is often insufficient because the most important information in a catalog is visual—styles, colors, accessories, and combinations of clothing items shown in images.

In this use case, a multimodal AI system allows users to ask natural language questions about apparel shown in a catalog. The system processes the catalog by extracting both the textual content and the images from each page. Vision-capable models analyze the images and generate structured descriptions of the clothing items, including style, colors, accessories, and possible use cases such as casual wear, sports fan apparel, or beach outfits. These descriptions are combined with any available page text to create a searchable representation of the catalog. The resulting multimodal records are embedded and stored in a vector database so that the system can retrieve the most relevant pages when a user asks a question.

When a user submits a question, the system searches the catalog using semantic similarity rather than exact keywords. It retrieves the pages whose combined text and image descriptions best match the user’s query. A language model then generates an answer based only on the retrieved content, often referencing the relevant page numbers. This allows users to interact with the catalog conversationally, making it easier to discover apparel styles, compare outfits, and identify items that match a specific need or theme.

This type of system is useful not only for customers browsing a catalog but also for internal retail teams. Merchandising teams can quickly identify styles across collections, marketing teams can locate outfits suitable for campaigns, and sales associates can assist customers more efficiently. The same multimodal pipeline approach can also be applied to other industries where visual information is critical, such as product catalogs, manufacturing inspection reports, maintenance manuals, and engineering documentation.

---

# Learning Goals

After studying this demo, students should understand:

- How multimodal data pipelines ingest and organize heterogeneous data
- How preprocessing decisions affect downstream AI systems
- Why derived artifacts (such as image descriptions) should be tracked
- How multimodal records can be stored for retrieval systems
- Why **data alignment** is a critical step in multimodal pipelines
- Why **tokenization at ingestion time may not always be the correct design choice**

---

# Pipeline Overview

The pipeline implemented in this demo follows these stages:

```
PDF Source
│
├── Text Extraction
│
├── Image Extraction
│
├── Image Preprocessing
│
├── Vision Model Description
│
├── Multimodal Record Construction
│
├── Embedding Generation
│
└── Vector Storage (Chroma)
```

Users can then ask questions about the catalog using a **retrieval-augmented chatbot**.

---

# Important Design Observations for Students

The purpose of this demo is not only to show code, but to highlight **design decisions** that occur when building real multimodal systems.

Students should pay attention to the following.

---

# 1. Missing Modality Step (Intentional Simplification)

The current pipeline processes:

- extracted **text**
- extracted **images**
- **LLM-generated descriptions**

However, many real-world multimodal pipelines also include additional modalities such as:

- tables
- sensor data
- structured metadata
- OCR text embedded in images
- timestamps from external systems

### Exercise / Design Question

**Add a missing modality step.**

Possible extensions include:

- table extraction from the PDF
- OCR text extraction from images
- product metadata ingestion
- style classification tags

Students should consider:

- how the new modality would be extracted
- how it should be aligned with existing data
- how it should be stored in the final record

---

# 2. Multimodal Alignment Step

The pipeline combines multiple data sources into a **single multimodal document record**.

Currently, the alignment is simple because the source is a PDF page.  
Each page acts as the **alignment key**.

However, real systems require more explicit alignment logic.

Students should consider expanding the alignment step by defining:

### Join Strategy

Examples:

```
page_number
document_id
timestamp
product_id
inspection_id
```

### Timestamp Tolerance

In time-series multimodal pipelines, signals may arrive at slightly different times.

Example:

```
image_timestamp = 10:05:02
sensor_timestamp = 10:05:04
tolerance = ±5 seconds
```

### Output Record Format

Example structured record:

```json
{
  "record_id": "",
  "page_number": "",
  "text": "",
  "image_path": "",
  "image_description": "",
  "preprocessing_metadata": {}
}
```

### Validation

Important checks include:

- missing modalities
- duplicate alignments
- corrupted images
- empty text fields

Students should think about **how multimodal records are validated before indexing**.

---

# 3. Multimodal Data Loader (Suggested Extension)

The demo currently extracts and processes data directly from the PDF.

In larger systems, a **multimodal data loader** is typically responsible for:

- reading source files
- detecting modality types
- dispatching to appropriate preprocessors
- returning normalized records

Example responsibilities:

```
MultimodalLoader
├── load_text()
├── load_images()
├── load_tables()
└── assemble_records()
```

Adding a loader abstraction would make the pipeline:

- easier to extend
- reusable across datasets
- easier to test

Students should consider how to design a **generic multimodal loader class**.

---

# 4. Image Preprocessing Decisions

The pipeline includes an explicit image preprocessing step:

```
preprocess_image()
```

Students should pay attention to the specific choices made:

### Dimensions

Images are resized to:

```
224 x 224
```

This size is commonly used in many computer vision models.

### Normalization

Pixel values are normalized:

```
pixel_value / 255.0
```

This converts values from:

```
0–255 → 0–1
```

### Data Format

Two representations are tracked:

```
HWC (height, width, channels)
CHW (channels, height, width)
```

Many deep learning frameworks use **CHW tensor layouts**, while image libraries often use **HWC arrays**.

The preprocessing metadata is saved so that downstream systems know:

- what transformations were applied
- how the image is represented

---

# 5. Why Image Tokenization Is Not Done Here

Students may notice that **image tokenization is intentionally NOT performed during ingestion**.

Instead, the system stores:

- raw images
- preprocessing metadata
- text descriptions

### Why?

Tokenization is typically **model-specific**.

For example:

- Vision Transformers
- CLIP
- GPT-style multimodal models

Each may require a different tokenization process.

If tokenization occurs during ingestion:

- the stored representation becomes tied to one model
- switching models later becomes difficult
- reprocessing the dataset may be required

### Design Decision

This pipeline defers tokenization until **retrieval or model execution time**.

Benefits include:

- model flexibility
- easier pipeline evolution
- reusable stored data

Students should consider when tokenization should occur in real production pipelines.

---

# System Architecture

```
PDF
│
├─ Text Extraction (PyMuPDF)
│
├─ Image Extraction
│
├─ Image Preprocessing (NumPy + PIL)
│
├─ Vision LLM Description
│
├─ Multimodal Record Creation
│
├─ Embedding Generation (Azure OpenAI)
│
└─ Vector Storage (Chroma)
```

---

# Technologies Used

| Component | Technology |
|-----------|------------|
| PDF parsing | PyMuPDF |
| Image processing | PIL / NumPy |
| LLM access | Azure OpenAI |
| Vision analysis | GPT multimodal deployment |
| Embeddings | Azure OpenAI embeddings |
| Vector store | Chroma |
| Secrets | Azure Key Vault |

---

# Running the Demo

Place the catalog in:

```
data/catalog.pdf
```

Run the script:

```bash
python demo.py
```

The system will:

1. Authenticate with Azure Key Vault
2. Build the vector index (if it does not exist)
3. Launch an interactive catalog assistant

Example query:

```
Ask about the catalog: Which page shows a casual baseball-style outfit?
```

---

# Key Takeaway

Building multimodal GenAI systems is not only about models.  
It requires careful decisions about:

- **data ingestion**
- **modality alignment**
- **preprocessing**
- **artifact tracking**
- **storage format**
- **model coupling**

Understanding these decisions is essential for designing reliable **data pipelines for generative AI systems**.
