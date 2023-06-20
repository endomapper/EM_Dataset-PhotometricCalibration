#!/bin/bash

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATASET_DIR=$1

function usage()
{
    echo "Usage: ./$( basename $0 ) <path_to_dataset>"
    exit 1
}

if [ "$#" -ne 1 ]; then
    echo 'Invalid number of arguments. Provide the path to the dataset only.'
    usage
fi

if [ ! -d "${DATASET_DIR}" ]; then
    echo "'${DATASET_DIR}' is not a valid directory. Check the provided path."
    usage
fi

if [ ! -d "${DATASET_DIR}/Calibrations" ]; then
    echo "Dataset not found at '${DATASET_DIR}'. Check the provided path."
    usage
fi

cd ${DATASET_DIR}/Calibrations

# calibrate photometry 
for i in Endoscope_*; do python ${SCRIPT_DIR}/../calibration/test_hculb.py -p ${DATASET_DIR}/Calibrations -s ${i} ; done
# collect all output data to a CSV file
for i in Endoscope_*; do echo ${i},$(cat ${i}/${i}_output.txt | rev | cut -d' ' -f1 | rev | tr '\n' ','); done > calibrations_output.csv

cd ${CURRENT_DIR}