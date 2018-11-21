#!/usr/bin/env bash

TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`

declare -r CUB_DOWNLOAD_URL="http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"

usage() {
    echo "This script downloads and extracts the CUB 2011 dataset from the following URL:"
    echo "  - ${CUB_DOWNLOAD_URL}"
    echo ""
    echo "USAGE:"
    echo "\$ sh get_cub_dataset.sh /path/to/_scratch"
    echo ""
}

download() {
    local DOWNLOAD_URL="$1"
    local TARGET_DIR=$(cd "$2" && pwd)

    echo "[${TIMESTAMP}] Downloading ${DOWNLOAD_URL} to ${TARGET_DIR}"

    wget -q --show-progress "${DOWNLOAD_URL}" -P "${TARGET_DIR}"
}

untar_and_cleanup() {
    local TARGET_DIR=$(cd $(dirname "$1") && pwd)
    local TARGET_FILEPATH="$1"

    echo "[${TIMESTAMP}] Extracting data from $1"

    tar -C "${TARGET_DIR}" -xzf "${TARGET_FILEPATH}"
    rm -f "${TARGET_FILEPATH}"
}

if [  $# -ne 1 ]; then
    usage
    echo "ERROR: need path to _scratch directory"
    echo ""
    exit 1
fi

if [[ ( $1 == "--help") ||  $1 == "-h" ]]; then
    usage
    exit 0
fi

if [ ! -d "$1" ]; then
    usage
    echo "ERROR: $1 does not exist"
    echo ""
    exit 1
fi

declare -r SCRATCH_DIR=$(cd "$1" && pwd)

DOWNLOAD_TARGET_DIR="${SCRATCH_DIR}/CUB_200_2011"
DOWNLOAD_FILENAME=$(basename "${CUB_DOWNLOAD_URL}")

mkdir -p "${DOWNLOAD_TARGET_DIR}"

download "${CUB_DOWNLOAD_URL}" "${DOWNLOAD_TARGET_DIR}"
untar_and_cleanup "${DOWNLOAD_TARGET_DIR}/${DOWNLOAD_FILENAME}"

echo "[${TIMESTAMP}] Finished downloading the CUB 2011 dataset"
