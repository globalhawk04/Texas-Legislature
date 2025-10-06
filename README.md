# Special Education Due Process Hearing Analysis

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![NLTK](https://img.shields.io/badge/NLTK-3.8-blue?logo=nltk)
![Pandas](https://img.shields.io/badge/Pandas-2.0-blue?logo=pandas)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

This project is a complete data engineering and Natural Language Processing (NLP) pipeline designed to analyze legal decisions from the Texas Education Agency's (TEA) special education due process hearings. The primary goal is to programmatically determine the outcome of each case—whether the "Petitioner" (typically a parent) or the "Respondent" (the school district) was the prevailing party.

The system automates the analysis of hundreds of dense, unstructured legal documents, providing a structured dataset for further legal and statistical analysis.

---

### Core Features

*   **Data Ingestion:** A resilient web scraper (`texas_due_process_extract.py`) navigates the TEA website, identifies links to PDF decision documents, and extracts the full text from each one using `PyPDF2`.
*   **Intelligent Text Extraction:** The scraper is optimized to only extract text from the final, most relevant pages of each PDF, where the "Conclusions of Law" and "Orders" are typically found.
*   **Rule-Based NLP Engine:** A sophisticated analysis script (`examine_ed_data_2.py`) that acts as a domain-specific classifier. It uses a carefully curated set of weighted keywords and regular expressions to score each case.
*   **Advanced Heuristics:** The NLP engine goes beyond simple keyword matching. It uses:
    *   **Sectional Analysis:** It intelligently isolates the "Conclusions of Law" and "Orders" sections of each document, applying different scoring weights to each.
    *   **Stemming:** Uses NLTK's `PorterStemmer` to ensure variations of a word (e.g., "grant," "granted") are all treated the same.
    *   **Negation Handling:** Actively looks for negating words ("not," "failed to") to correctly interpret the context of a keyword (e.g., "did *not* fail to provide" is a win for the Respondent).
*   **Structured Output:** The final analysis is saved to a clean, structured `decision_analysis.csv` file, including the docket number, the determined winner, and the scores for each party.

---

### Project Pipeline

The project follows a clear, multi-step ETL process:

1.  **Extraction (`texas_due_process_extract.py`):** Scrapes the TEA website for links to PDF legal decisions from 2010-2024 and extracts the raw text into a JSON file (`XX_ed.json`).
2.  **Transformation & Analysis (`examine_ed_data_2.py`):**
    *   Loads the raw JSON data.
    *   For each document, it applies the rule-based NLP engine to clean the text, isolate key sections, and score the content based on stemmed, weighted keywords and negation logic.
    *   Determines a "winner" (Petitioner, Respondent, or Mixed) based on the final scores.
3.  **Loading:** Saves the structured results into `decision_analysis.csv`, ready for review and further analysis.

*(Note: `examine_ed_data.py` and `examine_ed_data_1.py` represent earlier, iterative versions of the final analysis script.)*

---

### How to Run

#### Prerequisites
- Python 3.x
- Required libraries: `requests`, `PyPDF2`, `beautifulsoup4`, `nltk`, `pandas`

Install with pip:
```bash
pip install requests PyPDF2 beautifulsoup4 nltk pandas
