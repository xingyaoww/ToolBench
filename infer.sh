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

    python toolbench/inference/qa_pipeline.py \
        --tool_root_dir data/toolenv/tools/ \
        --backbone_model $BACKBONE_MODEL \
        --openai_key $OPENAI_API_KEY \
        --max_observation_length 1024 \
        --method CoT@1 \
        --input_query_file $QUERY_FILE \
        --output_answer_file $OUTPUT_DIR \
        --toolbench_key $TOOLBENCH_KEY \
        --action_mode $ACTION_MODE

    # Convert 
    CONVERTED_OUTPUT_DIR=data/eval_outputs/$BACKBONE_MODEL/$ACTION_MODE/converted
    mkdir -p $CONVERTED_OUTPUT_DIR
    CONVERTED_ANSWER_PATH=${CONVERTED_OUTPUT_DIR}/$QUERY_FILENAME.json
    python toolbench/tooleval/convert_to_answer_format.py\
        --answer_dir ${OUTPUT_DIR} \
        --method CoT@1 \
        --output ${CONVERTED_ANSWER_PATH}

    PASS_RATE_SAVE_PATH=data/eval_outputs/$BACKBONE_MODEL/$ACTION_MODE/pass_rate_results
    mkdir -p $PASS_RATE_SAVE_PATH
    export OPENAI_KEY=$OPENAI_API_KEY  # can ignore POOL_FILE if you set OPENAI_KEY

    # It will automatically skip the results that have been evaluated
    python toolbench/tooleval/eval_pass_rate.py \
        --converted_answer_path ${CONVERTED_OUTPUT_DIR} \
        --save_path ${PASS_RATE_SAVE_PATH} \
        --test_ids data/test_query_ids/ \
        --max_eval_threads 20 \
        --evaluate_times 4
}


MODEL=chat_completion:gpt-3.5-turbo-16k-0613

for testset in G1; do
    for action_mode in code_as_action json_as_action; do
        infer \
            data/test_instruction/${testset}_instruction.json \
            $action_mode \
            $MODEL
    done
done

# # For G1, G2, G3 instructions
# for testset in G1 G2 G3; do
#     for action_mode in code_as_action json_as_action; do
#         infer \
#             data/test_instruction/${testset}_instruction.json \
#             $action_mode \
#             $MODEL
#     done
# done
