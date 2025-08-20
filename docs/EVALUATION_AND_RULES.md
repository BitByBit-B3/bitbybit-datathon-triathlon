# Evaluation and Rules - Tech-Triathlon 2025 Datathon

## Competition Overview

The Tech-Triathlon 2025 Datathon requires predictions for two related tasks:

1. **Task 1**: Service processing time prediction (minutes after staff start)
2. **Task 2**: Staffing level prediction (number of employees needed)

## Submission Requirements

### File Structure
```
TeamName_Datathon.zip/
├── task1_predictions.csv          # Required at root
├── task2_predictions.csv          # Required at root  
├── demo_video_link.txt            # Required at root
├── code/                          # Implementation code
└── docs/                          # Documentation
```

### Output Format

#### task1_predictions.csv
- **Columns**: `row_id`, `prediction`
- **prediction**: Integer minutes (processing time after staff start)
- **Range**: Must be ≥ 0, typically 0-480 minutes (8 hours max)

#### task2_predictions.csv  
- **Columns**: `row_id`, `prediction`
- **prediction**: Integer number of employees needed
- **Range**: Must be ≥ 1 (at least 1 employee required)

### Input Schemas

#### task1_test_inputs.csv
- `row_id`: Unique identifier
- `date`: Date in YYYY-MM-DD format
- `time`: Time in HH:MM 24-hour format
- `task_id`: Task identifier

#### task2_test_inputs.csv
- `row_id`: Unique identifier  
- `date`: Date in YYYY-MM-DD format
- `section_id`: Section identifier

## Training Data

### bookings_train.csv
- `booking_id`: Unique booking identifier
- `booking_created_time`: When booking was created
- `appointment_time`: Scheduled appointment time
- `task_id`: Associated task
- `document_uploaded`: Boolean flag
- `queue_number`: Queue position (nullable)
- `satisfaction_rating`: Customer satisfaction (nullable)
- `check_in_time`: When customer checked in
- `check_out_time`: When service completed

### tasks.csv
- `task_id`: Unique task identifier
- `task_name`: Name of the task (nullable)
- `section_id`: Associated section
- `section_name`: Name of the section (nullable)

### staffing_train.csv
- `date`: Date in YYYY-MM-DD format
- `section_id`: Section identifier
- `employees_on_duty`: Number of employees working
- `total_task_time_minutes`: Total minutes of work

## Target Variables

### Task 1 Target
**Expected Processing Time** = `check_out_time - check_in_time` (in minutes)

**Data Quality Rules**:
- Skip rows where either timestamp is missing/unparseable
- Remove negative durations (data errors)
- Cap at 8 hours (480 minutes) to remove outliers
- If completely unavailable, use fallback proxy with clear logging

### Task 2 Target
**Primary**: `employees_on_duty` from staffing_train.csv

**Fallback**: If `employees_on_duty` missing, estimate from `total_task_time_minutes`:
```
employees_estimate = total_task_time_minutes / median_minutes_per_employee
```
Where `median_minutes_per_employee` is derived from rows with both values.

## Evaluation Metrics

While specific metrics aren't disclosed, typical regression metrics likely include:
- **Mean Absolute Error (MAE)**
- **Root Mean Square Error (RMSE)**
- **Mean Absolute Percentage Error (MAPE)**

## Validation Requirements

### Technical Validation
- ✅ Exact column names: `row_id`, `prediction`
- ✅ Integer predictions ≥ specified minimums
- ✅ No missing/NaN values
- ✅ Row count matches test input count
- ✅ Consistent row_id ordering

### Business Logic Validation
- ✅ Task 1: Processing times are reasonable (0-480 minutes)
- ✅ Task 2: Staffing levels are realistic (1-50 employees per section)
- ✅ Predictions handle edge cases gracefully

## Best Practices

### Model Development
1. **Robust Baselines**: Start with statistical baselines (median by key features)
2. **Graceful Degradation**: Handle missing data with intelligent fallbacks
3. **Cross-Validation**: Use time-aware splits for temporal data
4. **Feature Engineering**: Extract time-based and categorical features safely

### Code Quality
1. **Error Handling**: Comprehensive try-catch blocks
2. **Logging**: Detailed logs for debugging and validation
3. **Modularity**: Separate preprocessing, training, and prediction
4. **Documentation**: Clear README and code comments

### Submission Checklist
- [ ] Both CSV files at root with correct format
- [ ] Demo video link provided
- [ ] Code runs without errors
- [ ] Predictions are within valid ranges
- [ ] No hardcoded paths or credentials
- [ ] Requirements.txt includes all dependencies

## Common Pitfalls to Avoid

1. **Wrong file locations**: CSVs must be at root, not in subdirectories
2. **Incorrect column names**: Must be exactly `row_id,prediction`
3. **Non-integer predictions**: All predictions must be integers
4. **Missing row handling**: Must produce predictions for ALL test rows
5. **Hardcoded assumptions**: Code should handle various data distributions