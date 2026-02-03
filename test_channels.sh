#!/bin/bash
# Test script for configurable input/output channels feature

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

# Step 3: Test custom output channels
echo ""
echo "Test 2: Custom output channels (1 channel only)"
echo "----------------------------------------"
python main.py -m stconvs2s-r -e 3 -b 10 -i 1 -dsp data/toy-dataset-seq5-ystep5.nc --small-dataset --output-channels 1 --verbose --no-stop 2>&1 | tee test2_output_channels.log

# Step 4: Test custom output channels (2 channels)
echo ""
echo "Test 3: Custom output channels (2 channels)"
echo "----------------------------------------"
python main.py -m stconvs2s-r -e 3 -b 10 -i 1 -dsp data/toy-dataset-seq5-ystep5.nc --small-dataset --output-channels 2 --verbose --no-stop 2>&1 | tee test3_output_channels_2.log

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""
echo "Check the log files for details:"
echo "  - test1_default.log"
echo "  - test2_output_channels.log"
echo "  - test3_output_channels_2.log"
echo ""
echo "Look for '[X_train] Shape:' and '[y_train] Shape:' in the logs"
echo "to verify the channel dimensions are correct."
echo ""
echo "Expected shapes:"
echo "  Test 1: y should have 5 channels (dim 1) - all channels"
echo "  Test 2: y should have 1 channel (dim 1) - only first channel"
echo "  Test 3: y should have 2 channels (dim 1) - first 2 channels"
