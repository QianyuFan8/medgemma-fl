# Federated Fine-Tuning of MedGemma and Gemma Scope 2 Interpretability
> **Coordinated via NVIDIA FLARE with LoRA/QLoRA Adapters for Pediatric Oncology Clinical Trial Matching**

## Overview

This capstone implements a working slice of the **CCDI Federated Learning Initiative**: a federated learning pipeline in which simulated pediatric oncology sites locally fine-tune MedGemma using LoRA/QLoRA adapters while keeping raw patient data local.

**NVIDIA FLARE** coordinates federated training and aggregation of adapter updates across sites. **Gemma Scope 2** is integrated as an interpretability and auditing layer to examine model behavior across federated rounds and during downstream clinical trial matching.

The resulting MedGemma + global CCDI adapter is evaluated for extracting patient eligibility information and matching patients against pediatric oncology clinical trial criteria.

**Primary use case:** AI-assisted clinical trial matching for pediatric oncology patients.
