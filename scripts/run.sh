#!/bin/bash

# Create a backup of the original config.yaml
cp config.yaml config.yaml.backup

# Function to update config.yaml
update_config() {
    local agent=$1
    local interaction=$2
    local features=$3
    local extra_features=$4
    local filter_nonzero=$5
    local output_file=$6
    
    # Create a temporary config file
    cat > config_temp.yaml << EOF
# DCM Model Configuration

data:
  data_root: "data/features/biz"
  agent: "${agent}"
  block: "blocks_fine"
  agent_filter_dict: null
  block_filter_dict: null
  filter_nonzero_features: ${filter_nonzero}

model:
  model_type: "interactions"
  distance_interaction: "l2_log"
  race_interaction: null
  income_interaction: null
  distance_features_interaction: "l2_log_product"
  interaction: ${interaction}
  include_extra_features: ${extra_features}

  # Control features (not interacted with distance)
  control_names:
    - "log_consumer_control"
    - "log_white_collar_control"
    - "log_blue_collar_control"
EOF

    # Add control_names and feature_names based on the features type
    if [ "$features" = "biz" ]; then
        cat >> config_temp.yaml << EOF
  
  # Features to use in the model (will be interacted with distance if interaction=true)
  feature_names:
    - "log_consumer"
    - "log_white_collar"
    - "log_blue_collar"
    - "consumer_hetero"
EOF
    elif [ "$features" = "full_biz" ]; then
        cat >> config_temp.yaml << EOF
  
  # Features to use in the model (only controls for full version)
  feature_names: []
EOF
    elif [ "$features" = "lnemp" ]; then
        cat >> config_temp.yaml << EOF
  
  # Features to use in the model (controls + lnemp features)
  feature_names:
    - "log_emp_consumer"
    - "log_emp_white_collar"
    - "log_emp_blue_collar"
    - "log_emp_consumer_hetero"
EOF
    fi

    # Add optimizer section
    cat >> config_temp.yaml << EOF

optimizer:
  chunk_size: 2048
  max_iter: 1000

output_file: "${output_file}"
EOF

    # Replace the actual config.yaml
    mv config_temp.yaml config.yaml
}

# Function to run main.py with error handling
run_model() {
    local description=$1
    echo "=========================================="
    echo "Running: $description"
    echo "=========================================="
    
    python main.py --config config.yaml
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed: $description"
    else
        echo "✗ Failed: $description"
    fi
    echo ""
}

# Generate all configurations and run

# 1. offenders_biz.json - no interaction, with controls
update_config "offenders_fine" "false" "biz" "false" "false" "data/estimators/biz/offenders_biz.json"
run_model "offenders_biz.json"

# 2. offenders_biz_int.json - with interaction, with controls
update_config "offenders_fine" "true" "biz" "false" "false" "data/estimators/biz/offenders_biz_int.json"
run_model "offenders_biz_int.json"

# 3. offenders_full_biz.json - no interaction, full version with extra features
update_config "offenders_fine" "false" "full_biz" "true" "false" "data/estimators/biz/offenders_full_biz.json"
run_model "offenders_full_biz.json"

# 4. offenders_full_biz_int.json - with interaction, full version with extra features
update_config "offenders_fine" "true" "full_biz" "true" "false" "data/estimators/biz/offenders_full_biz_int.json"
run_model "offenders_full_biz_int.json"

# 5. offenders_lnemp.json - no interaction, lnemp features
update_config "offenders_fine" "false" "lnemp" "false" "false" "data/estimators/biz/offenders_lnemp.json"
run_model "offenders_lnemp.json"

# 6. offenders_lnemp_int.json - with interaction, lnemp features
update_config "offenders_fine" "true" "lnemp" "false" "false" "data/estimators/biz/offenders_lnemp_int.json"
run_model "offenders_lnemp_int.json"

# Now run for victims

# 7. victims_biz.json - no interaction, with combined features
update_config "victims_fine" "false" "biz" "false" "false" "data/estimators/biz/victims_biz.json"
run_model "victims_biz.json"

# 8. victims_biz_int.json - with interaction, with combined features
update_config "victims_fine" "true" "biz" "false" "false" "data/estimators/biz/victims_biz_int.json"
run_model "victims_biz_int.json"

# 9. victims_full_biz.json - no interaction, full version with extra features
update_config "victims_fine" "false" "full_biz" "true" "false" "data/estimators/biz/victims_full_biz.json"
run_model "victims_full_biz.json"

# 10. victims_full_biz_int.json - with interaction, full version with extra features
update_config "victims_fine" "true" "full_biz" "true" "false" "data/estimators/biz/victims_full_biz_int.json"
run_model "victims_full_biz_int.json"

# 11. victims_lnemp.json - no interaction, lnemp features
update_config "victims_fine" "false" "lnemp" "false" "false" "data/estimators/biz/victims_lnemp.json"
run_model "victims_lnemp.json"

# 12. victims_lnemp_int.json - with interaction, lnemp features
update_config "victims_fine" "true" "lnemp" "false" "false" "data/estimators/biz/victims_lnemp_int.json"
run_model "victims_lnemp_int.json"

# Restore original config.yaml
cp config.yaml.backup config.yaml
rm config.yaml.backup

echo "=========================================="
echo "All model runs completed!"
echo "=========================================="