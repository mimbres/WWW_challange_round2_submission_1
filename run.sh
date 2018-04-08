
#!/bin/bash

# This script expects that the following environment variables are set
# when it is being executed:
#
# TEST_DIRECTORY: directory containing all the test mp3 files
# OUTPUT_PATH: path where the output CSV file will be written


# SET Data Path
#export TEST_DIRECTORY=data/crowdai_fma_test/
export FEATURE_DIRECTORY=data/preprocessed_data_spec/test/
#export OUTPUT_PATH=output.csv


echo "TEST Directory: $TEST_DIRECTORY"
echo "FEATURE Directory: $FEATURE_DIRECTORY"
echo "OUTPUT PATH: $OUTPUT_PATH"


# STEP 1: Preprocessing Audio : You can skip prerocessing later.
python3 util/audio_processing_test_np_sep_spec_MlutiThread.py --fma_testset_filedir $TEST_DIRECTORY --out_filedir $FEATURE_DIRECTORY

# STEP 2: Test
python3 test_exp15.py --feature_dir=$FEATURE_DIRECTORY --output_path=$OUTPUT_PATH --load "checkpoints/exp15/checkpoint_115.pth.tar"
