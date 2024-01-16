#!/bin/bash
export PYTHONPATH=./

TOOLBENCH_KEY=w9XsUZYhnLdxZqxDdr8w6xCe4e1lKsyY0CHvHbZsTtDO9NcGfP

# pack above script into a function that takes in the input query file, action mode, and backbone model as arguments
function infer() {
    QUERY_FILE=$1
    # get the filename without extension
    QUERY_FILENAME=$(basename -- "$QUERY_FILE")
    # remove the extension
    QUERY_FILENAME="${QUERY_FILENAME%.*}"
    ACTION_MODE=$2
    # assert action mode is either code_as_action or json_as_action
    if [ "$ACTION_MODE" != "code_as_action" ] && [ "$ACTION_MODE" != "json_as_action" ]; then
        echo "Action mode must be either code_as_action or json_as_action"
        exit 1
    fi
    BACKBONE_MODEL=$3  # e.g. chat_completion:gpt-3.5-turbo-16k-0613
    MODEL_NAME=$(echo $BACKBONE_MODEL | cut -d':' -f2)  # e.g. gpt-3.5-turbo-16k-0613
    
    OUTPUT_DIR=data/eval_outputs/$BACKBONE_MODEL/$ACTION_MODE/$QUERY_FILENAME
    mkdir -p $OUTPUT_DIR
    echo "Output directory: $OUTPUT_DIR"

    # If ls $OUTPUT_DIR has no files, skip
    if [ -z "$(ls -A $OUTPUT_DIR)" ]; then
        echo "Directory $OUTPUT_DIR is empty. Skipping..."
        return
    fi

    # Convert 
    CONVERTED_OUTPUT_DIR=${OUTPUT_DIR}.converted
    mkdir -p $CONVERTED_OUTPUT_DIR
    python toolbench/tooleval/convert_to_answer_format.py\
        --answer_dir ${OUTPUT_DIR} \
        --method CoT@1 \
        --output ${CONVERTED_OUTPUT_DIR}/answer.json
}

infer \
    data/test_instruction/G1_instruction.json \
    code_as_action \
    chat_completion:gpt-3.5-turbo-16k-0613

infer \
    data/test_instruction/G1_instruction.json \
    json_as_action \
    chat_completion:gpt-3.5-turbo-16k-0613
