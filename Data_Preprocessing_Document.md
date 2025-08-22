# Data Pre-Processing Document
## BitByBit Datathon 2025 - Team BitByBit

### Overview
This document outlines the data cleaning, feature engineering, and preprocessing pipeline implemented for both Task 1 (Service Processing Time Prediction) and Task 2 (Staffing Requirements Prediction).

## 1. Data Cleaning

### Bookings Dataset Cleaning
**Target Variable Creation (Task 1)**:
- Calculated processing time as `check_out_time - check_in_time`
- Filtered valid processing times: 5.9 to 217.6 minutes (removed outliers >8 hours)
- Resulted in 197,601 valid samples from 203,693 raw bookings

**DateTime Parsing**:
- Standardized all datetime columns with robust parsing fallbacks
- Handled missing timestamps gracefully (6,092 missing check-in/out times)
- Applied UTC timezone normalization for consistency

**Data Quality Filters**:
- Removed negative processing times (invalid data)
- Capped maximum processing time at 480 minutes (8-hour business limit)
- Validated appointment time formats and ranges

### Staffing Dataset Cleaning
**Employee Count Validation**:
- Ensured minimum 1 employee per section (business rule)
- Capped maximum at 50 employees per section (outlier removal)
- Validated 5,802 staffing records across 6 sections

**Task Time Validation**:
- Filtered total_task_time_minutes for realistic ranges (91-10,658 minutes)
- Calculated median minutes per employee: 370.9 (reasonable 6-hour productive work)

### Tasks Dataset Processing
**Section Mapping**:
- Defined 6 business-logical sections:
  - SEC-001: Identification Services
  - SEC-002: Vital Records  
  - SEC-003: Business Registration
  - SEC-004: Revenue Collection
  - SEC-005: Social Services
  - SEC-006: Information Services
- Mapped 19 tasks across sections with meaningful names

## 2. Feature Engineering

### Task 1 Features (Processing Time Prediction)
**Temporal Features**:
- `appt_hour`: Hour of appointment (0-23) - captures staff efficiency patterns
- `appt_weekday`: Day of week (0-6) - Monday=0, captures weekly patterns  
- `appt_month`: Month (1-12) - seasonal demand variations
- `is_weekend`: Boolean weekend indicator - different staffing on weekends

**Task-Related Features**:
- `task_id`: Direct task identifier - primary driver of processing complexity
- `section_id`: Derived from task mapping - department-level patterns
- `num_documents`: Document count - complexity indicator

**Derived Features**:
- Time categories (morning/afternoon/evening) for broader temporal patterns
- Task complexity scores based on historical processing times

### Task 2 Features (Staffing Prediction)
**Temporal Features**:
- `weekday`: Day of week - primary staffing pattern driver
- `month`: Monthly patterns - seasonal demand
- `quarter`: Quarterly business cycles
- `is_weekend`: Weekend vs weekday staffing models

**Section Features**:
- `section_id`: Department identifier - baseline staffing requirements
- Historical workload patterns per section

## 3. Preprocessing Rationale

### Statistical Approach Choice
**Why Median-Based Baselines**:
- More robust to outliers than mean-based approaches
- Government service times often have skewed distributions
- Reliable predictions even with limited training data
- Interpretable results for business stakeholders

### Hierarchical Fallback Strategy
**Task 1 Fallback Levels** (7 levels):
1. `task_id + appt_hour + appt_weekday` (most specific)
2. `task_id + appt_hour` 
3. `task_id + appt_weekday`
4. `task_id` only
5. `appt_hour + appt_weekday`
6. `appt_hour` only  
7. Global median (final fallback)

**Task 2 Fallback Levels** (4 levels):
1. `section_id + weekday` (most specific)
2. `section_id` only
3. `weekday` only
4. Global median (final fallback)

**Rationale**: Ensures 100% test coverage while maximizing prediction specificity

### Missing Data Strategy
**Approach**: Progressive degradation rather than imputation
- Use available features at each fallback level
- Skip levels with missing required features
- Always provide prediction via global statistics

**Reasoning**: Maintains data integrity, avoids introducing bias through imputation

## 4. Validation and Quality Assurance

### Output Validation
**Range Checking**:
- Task 1: 0-480 minutes (business hours constraint)
- Task 2: 1-50 employees (operational limits)

**Format Compliance**:
- Integer outputs only (competition requirement)
- Proper CSV schema validation
- No missing predictions (100% coverage guaranteed)

### Cross-Validation Strategy
**Time-Aware Splitting**: Prevents data leakage from future to past
**Holdout Validation**: 20% of training data reserved for validation
**Robustness Testing**: Verified performance on edge cases and missing data scenarios

## 5. Performance Metrics

### Training Results
**Task 1 Performance**:
- MAE: 14.37 minutes
- RMSE: 19.74 minutes
- Training samples: 197,601
- Coverage: 100% fallback success

**Task 2 Performance**:
- MAE: 0.60 employees
- RMSE: 0.80 employees  
- Training samples: 5,802
- Coverage: 100% fallback success

### Prediction Distribution
**Task 1**: Range 23-74 minutes, median-centered around business norms
**Task 2**: Range 2-9 employees per section, reflecting operational requirements

## 6. Technical Implementation

### Code Architecture
- Modular design with separate preprocessing, feature extraction, and prediction modules
- Robust error handling with comprehensive logging
- Safe I/O operations with automatic data type validation
- Reproducible pipeline with deterministic outputs

### Dependencies
- Standard Python data science stack (pandas, numpy, scikit-learn)
- No proprietary models or APIs used (competition compliant)
- Optional advanced models available but defaulting to robust baselines

---

**Conclusion**: The preprocessing pipeline balances accuracy with reliability, ensuring robust predictions across all test scenarios while maintaining interpretability and business logic compliance.