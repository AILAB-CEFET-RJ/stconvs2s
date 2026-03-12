#!/bin/bash
# Test script for configurable output channels feature

GREEN="\033[0;32m"
RED="\033[0;31m"
NC="\033[0m"

log_status() {
    local exit_code=$1
    local test_name=$2
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✔ $test_name passed (exit code: $exit_code)${NC}"
    else
        echo -e "${RED}✘ $test_name failed (exit code: $exit_code)${NC}"
    fi
}

echo "=========================================="
echo "Testing Configurable Channels Feature"
echo "=========================================="
echo ""

# Step 1: Create toy dataset
echo "Step 1: Creating toy dataset..."
python create_toy_dataset.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to create toy dataset"
    exit 1
fi

echo ""
echo "=========================================="
echo "Running Tests"
echo "=========================================="

# Step 2: Test default behavior (backward compatibility)
echo ""
echo "Test 1: Default behavior (all output channels)"
echo "----------------------------------------"
python main.py -m stconvs2s-r -e 3 -b 10 -i 1 -dsp data/toy-dataset-seq5-ystep5.nc --small-dataset --verbose --no-stop 2>&1 | tee test1_default.log
log_status ${PIPESTATUS[0]} "Test 1"

# Step 3: Test custom output channels
echo ""
echo "Test 2: Custom output channels (1 channel only)"
echo "----------------------------------------"
python main.py -m stconvs2s-r -e 3 -b 10 -i 1 -dsp data/toy-dataset-seq5-ystep5.nc --small-dataset --output-channels 1 --verbose --no-stop 2>&1 | tee test2_output_channels.log
log_status ${PIPESTATUS[0]} "Test 2"

# Step 4: Test custom output channels (2 channels)
echo ""
echo "Test 3: Custom output channels (2 channels)"
echo "----------------------------------------"
python main.py -m stconvs2s-r -e 3 -b 10 -i 1 -dsp data/toy-dataset-seq5-ystep5.nc --small-dataset --output-channels 2 --verbose --no-stop 2>&1 | tee test3_output_channels_2.log
log_status ${PIPESTATUS[0]} "Test 3"

echo ""
echo "=========================================="
echo "Optional Tests (full dataset files)"
echo "=========================================="

echo ""
echo "Test 4a: CHIRPS dataset (ystep5) with output-channels 1"
echo "----------------------------------------"
DATA_FILE="data/dataset-chirps-1981-2019-seq5-ystep5.nc"
if [ -f "$DATA_FILE" ]; then
    echo "✓ File exists: $DATA_FILE"
    python main.py -m stconvs2s-c -e 1 -b 4 -i 1 -dsp "$DATA_FILE" --small-dataset --output-channels 1 --verbose --no-stop 2>&1 | tee test4a_chirps_ystep5_ch1.log
    log_status ${PIPESTATUS[0]} "Test 4a"
else
    echo "File not found: $DATA_FILE - skipping test"
fi

echo ""
echo "Test 5a: UCAR dataset (ystep5) with output-channels 1"
echo "----------------------------------------"
DATA_FILE="data/dataset-ucar-1979-2015-seq5-ystep5.nc"
if [ -f "$DATA_FILE" ]; then
    echo "✓ File exists: $DATA_FILE"
    python main.py -m stconvs2s-r -e 1 -b 4 -i 1 -dsp "$DATA_FILE" --small-dataset --output-channels 1 --verbose --no-stop 2>&1 | tee test5a_ucar_ystep5_ch1.log
    log_status ${PIPESTATUS[0]} "Test 5a"
else
    echo "File not found: $DATA_FILE - skipping test"
fi

echo ""
echo "Test 6a: CHIRPS dataset (ystep15) with output-channels 1"
echo "----------------------------------------"
DATA_FILE="data/dataset-chirps-1981-2019-seq5-ystep15.nc"
if [ -f "$DATA_FILE" ]; then
    echo "✓ File exists: $DATA_FILE"
    python main.py -m stconvs2s-c -e 1 -b 4 -i 1 -dsp "$DATA_FILE" --small-dataset --step 15 --output-channels 1 --verbose --no-stop 2>&1 | tee test6a_chirps_ystep15_ch1.log
    log_status ${PIPESTATUS[0]} "Test 6a"
else
    echo "File not found: $DATA_FILE - skipping test"
fi

echo ""
echo "Test 7a: UCAR dataset (ystep15) with output-channels 1"
echo "----------------------------------------"
DATA_FILE="data/dataset-ucar-1979-2015-seq5-ystep15.nc"
if [ -f "$DATA_FILE" ]; then
    echo "✓ File exists: $DATA_FILE"
    python main.py -m stconvs2s-r -e 1 -b 4 -i 1 -dsp "$DATA_FILE" --small-dataset --step 15 --output-channels 1 --verbose --no-stop 2>&1 | tee test7a_ucar_ystep15_ch1.log
    log_status ${PIPESTATUS[0]} "Test 7a"
else
    echo "File not found: $DATA_FILE - skipping test"
fi

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""
echo "Check the log files for details:"
echo "  - test1_default.log"
echo "  - test2_output_channels.log"
echo "  - test3_output_channels_2.log"
echo "  - test4a_chirps_ystep5_ch1.log   (optional - if dataset present)"
echo "  - test5a_ucar_ystep5_ch1.log     (optional - if dataset present)"
echo "  - test6a_chirps_ystep15_ch1.log  (optional - if dataset present)"
echo "  - test7a_ucar_ystep15_ch1.log    (optional - if dataset present)"
echo ""
echo "Look for '[X_train] Shape:' and '[y_train] Shape:' in the logs"
echo "to verify the channel dimensions are correct."
echo ""
echo "Expected shapes:"
echo "  Test 1: y should have 5 channels (dim 1) - all channels"
echo "  Test 2: y should have 1 channel (dim 1) - only first channel"
echo "  Test 3: y should have 2 channels (dim 1) - first 2 channels"
echo "  Tests 4a/5a: y should have 1 channel, 5 time steps"
echo "  Tests 6a/7a: y should have 1 channel, 15 time steps (--step 15)"
