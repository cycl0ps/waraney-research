# Dataset Catalog

Daftar seluruh dataset yang digunakan dalam penelitian dan eksperimen.

---

# Dataset Summary

| Nama Dataset | File Dataset                        | Alias | Keterangan                        |
| ------------ | ----------------------------------- | ----- | --------------------------------- |
| PEX01 400a   | `data/processed/dataset_pex01_400a` | 400a  | Dataset yang digunakan pada PEX01 |

---

# Dataset Overview

## PEX01 400a

### Overview

- Dataset digunakan sebagai testing set pada eksperimen PEX01. Data berasal dari abstrak artikel jurnal Unsrat.
- Teks AI dihasilkan dari teks Human menggunakan model GPT-4.1-mini.
- Dataset memiliki distribusi seimbang antara Human dan AI serta antara Bahasa Indonesia dan Bahasa Inggris.
- Dataset bersifat frozen dan tidak diubah setelah eksperimen dilakukan.

### Details

| Field      | Value               |
| ---------- | ------------------- |
| Dataset ID | PEX01-400A          |
| Alias      | 400a                |
| Type       | Testing Dataset     |
| Format     | CSV                 |
| Language   | Indonesian, English |
| Domain     | Academic Abstract   |
| Used In    | PEX01               |

### Composition

| Label | Count |
| ----- | ----: |
| Human |   200 |
| AI    |   200 |
| Total |   400 |

### Features

| Column     | Description                |
| ---------- | -------------------------- |
| doc_id     | Unique document identifier |
| lang       | Document language          |
| year       | Publication year           |
| word_count | Number of words            |
| text       | Document text              |
| label      | Class label (Human / AI)   |
| source     | Source of document         |

### Notes
