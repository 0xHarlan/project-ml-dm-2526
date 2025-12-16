# Project Machine Learning & Data Mining - 2025/2026

This repository contains the source code for my Machine Learning & Data Mining course, year 2025/2026.

## Overview

> ⚠️ **Disclaimer**  
> This code was developped in an educational way. It has no vocation to be maintained, improved or reused.  
> It serves only as presentation support.  

## Context

Goal is to predict the presence of various gases using a dataset of sensors values.

* **Link to the challenge:** [Identification de gaz toxiques par Bertin Technologies ](https://challengedata.ens.fr/challenges/156)

## Project structure

```text
.
├── datasets/            
├── src/                
│   ├── features.py      # Feature Engineering 
│   ├── models.py        # Models factory
│   ├── metrics.py       # Challenge metric (provided, rmse)
│   ├── utils.py         
│   └── graphs.py        
├── main.py              # Orchestration script
├── requirements.txt     # Dependencies
└── README.md 
```