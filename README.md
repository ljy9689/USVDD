This repository contains the implementation of the Uncertain Support Vector Data Description (USVDD) method, designed for outlier detection in environments with uncertain observational data. The USVDD method is specifically tailored for scenarios where data uncertainty arises from inherent noise, missing values, or other sources of imprecision.

Contents

This repository includes:
USVDD Method: The core implementation of the proposed Uncertain Support Vector Data Description (USVDD) algorithm, designed to handle aleatory uncertainty in high-dimensional datasets and enable one-class classification when fault samples are limited.

12 Datasets: Training and testing datasets from 12 real-world applications. These datasets are used to validate the performance of the USVDD method and compare it against other state-of-the-art methods for outlier detection.

Comparison Methods: Implementation of 4 alternative methods for handling uncertain data and outlier detection, used for comparative analysis with USVDD. These methods include traditional outlier detection techniques as well as uncertainty-aware models.

Key Features

Aleatory Uncertainty Handling: The USVDD method incorporates uncertainty theory to handle data with aleatory uncertainty, making it suitable for real-world datasets with noisy or imprecise data.

One-Class Classification: USVDD is designed for one-class classification tasks, which is crucial for detecting anomalies in systems where faulty samples are rare.

Publicly Available Datasets: This repository provides easy access to the 12 datasets used for validation. Researchers can use them to test their methods or compare performance against USVDD.
