#!/bin/bash
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
ERROR="[${RED}ERROR${NC}]"
INFO="[${BLUE}INFO${NC}]"
WARNING="[${YELLOW}WARNING${NC}]"
OK="[${GREEN}OK${NC}]"

# Example:
# $> ./calibrate.sh HCULB_03030 small

SEQUENCE=${1}
PATTERN_PRESET="${2:big}"
CAMERA_MODEL="${3:kb4}"
case $PATTERN_PRESET in
    "small")
        GRID_SMALL_RAD='0.0008470806985280002'
        GRID_LARGE_RAD='0.0012661312207680002'
        GRID_SPACING='0.0042238'
        ;;
    "big")
        GRID_SMALL_RAD='0.0010588508731600002'
        GRID_LARGE_RAD='0.0015826640259600002'
        GRID_SPACING='0.0052797'
        ;;
    *)
        echo -e "${ERROR} Invalid pattern preset '${PATTERN_PRESET}'"
        exit 1
esac

echo -e "${INFO} Using sequence ${SEQUENCE}..."

if [ -d "${SEQUENCE}/${SEQUENCE}_frames" ]; then
    echo -e "${WARNING} Frame files already exist. Skipping frame extraction."
else
    echo -e "${INFO} Extracting frames..."
    [ -f "${SEQUENCE}/${SEQUENCE}.mov" ] && VIDEO_FILE="${SEQUENCE}/${SEQUENCE}.mov"
    [ -f "${SEQUENCE}/${SEQUENCE}_lossless.mov" ] && VIDEO_FILE="${SEQUENCE}/${SEQUENCE}_lossless.mov"
    if [ -z ${VIDEO_FILE+x} ]; then 
        echo -e "${ERROR} Calibration video not found."
        exit 2
    fi
    echo -e "${INFO} Using '${VIDEO_FILE}'..."
    mkdir "${SEQUENCE}/${SEQUENCE}_frames"
    FRAMES="${SEQUENCE}/${SEQUENCE}_frames/%06d.png"
    mkdir "${SEQUENCE}/ffmpeg_logs"
    ffmpeg -r 1 -i ${VIDEO_FILE} -r 1 ${FRAMES} &> "${SEQUENCE}/ffmpeg_logs/stdout.txt"
    echo -e "${OK} Frames extracted to '${SEQUENCE}/${SEQUENCE}_frames'"
fi

if [ -f "${SEQUENCE}/${SEQUENCE}_geometrical.xml" ]; then
    echo -e "${OK} Geometric calibration file found."
else
    echo -e "${WARNING} Geometric calibration file not found. This script will calibrate the geometric parameters. Please note that it is preferable to perform the photometric calibration as indicated in the 'Geometric calibration' section of the README."
    sleep 3
    echo -e "${INFO} Calibrating endoscope..."
    mkdir -p "${SEQUENCE}/vicalib_logs"

    vicalib \
    -grid_preset small \
    -grid_small_rad ${GRID_SMALL_RAD} \
    -grid_large_rad ${GRID_LARGE_RAD} \
    -grid_spacing ${GRID_SPACING} \
    -max_iters 50000 \
    -models $CAMERA_MODEL \
    -frame_skip 25 \
    -cam "file://${SEQUENCE}/${SEQUENCE}_frames/"'*.png' \
    -output "${SEQUENCE}/${SEQUENCE}_geometrical.xml" \
    -calibrate_intrinsics \
    -log_dir "${SEQUENCE}/vicalib_logs" &> "${SEQUENCE}/vicalib_logs/${SEQUENCE}_inoutliers.txt"

    echo -e "${OK} Calibration saved at '${SEQUENCE}/${SEQUENCE}_geometrical.xml'"
fi

if [ -f "${SEQUENCE}/${SEQUENCE}_poses.csv" ]; then
    echo -e "${WARNING} Poses file already exists. Skipping pose estimation."
else
    echo -e "${INFO} Recovering camera poses..."
    mkdir -p "${SEQUENCE}/vicalib_logs"

    vicalib \
    -grid_preset small \
    -grid_small_rad ${GRID_SMALL_RAD} \
    -grid_large_rad ${GRID_LARGE_RAD} \
    -grid_spacing ${GRID_SPACING} \
    -max_iters 50000 \
    -models $CAMERA_MODEL \
    -frame_skip 0 \
    -cam "file://${SEQUENCE}/${SEQUENCE}_frames/"'*.png' \
    -has_initial_guess \
    -nocalibrate_intrinsics \
    -model_files "${SEQUENCE}/${SEQUENCE}_geometrical.xml" \
    -log_dir "${SEQUENCE}/vicalib_logs" &> "${SEQUENCE}/vicalib_logs/${SEQUENCE}_inoutliers.txt"

    mv poses.csv "${SEQUENCE}/${SEQUENCE}_poses.csv"
    rm poses.txt
    rm cameras.xml
    echo -e "${OK} Poses saved at '${SEQUENCE}/${SEQUENCE}_poses.csv'"
fi