build_run_simulations_cmd () {
    # Arguments: 
    #   1. step
    #   2. design
    #   3. methods_le
    #   4. methods_fse
    #   5. rotations
    #   6. regimes_fse
    
    # Parse arguments
    METHODS_LE="${3//,/ }"
    METHODS_FSE="${4//,/ }"
    ROTATIONS="${5//,/ }"
    REGIMES_FSE="${6//,/ }"

    # Build run_simulations command
    CMD="python scripts/run_simulations.py --config roar"
    CMD+=" --design $2"
    CMD+=" --steps $1"
    if [ ! -z "$METHODS_LE" ]; then
        CMD+=" --methods_le $METHODS_LE"
    fi
    if [ ! -z "$METHODS_FSE" ]; then
        CMD+=" --methods_fse $METHODS_FSE"
    fi
    if [ ! -z "$ROTATIONS" ]; then
        CMD+=" --rotations $ROTATIONS"
    fi
    if [ ! -z "$REGIMES_FSE" ]; then
        CMD+=" --regimes_fse $REGIMES_FSE"
    fi
    CMD+=" --benchmark"

    echo $CMD
}


