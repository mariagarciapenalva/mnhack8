#!/bin/bash

# module load nvidia-hpc-sdk/26.5
# module load python/3.14.7-gcc

CONFIG_FILE="config.json"

get() { jq -r --arg k "$1" '.[$k] // empty' <<< "$config"; }


while read -r config <&3; do

    TAG=$(get tag);       NP=$(get np); NP=${NP:-1}
    N=$(get N);           T_FINAL=$(get t_final);  DT=$(get dt)
    FIELD=$(get field);   OUTDIR=$(get outdir)

    if [[ -z "$N" || -z "$T_FINAL" || -z "$DT" || -z "$FIELD" || -z "$OUTDIR" ]]; then
        echo "[$TAG] N, t_final, dt, field and outdir are required -- skipping" >&2
        continue
    fi
    if [[ ! -f "${FIELD}_k.bin" || ! -f "${FIELD}_rhoc.bin" ]]; then
        echo "[$TAG] missing ${FIELD}_{k,rhoc}.bin (run scripts/gen_field.py) -- skipping" >&2
        continue
    fi
    mkdir -p "$OUTDIR"

    NSYS_ARGS=(
        -t cuda,nvtx,mpi
        --cuda-memory-usage=true
        --stats=true
        -f true
        -o "$OUTDIR/prof_$TAG"
    )

    APP_ARGS=("$N" "$T_FINAL" "$DT" "$FIELD" "$OUTDIR" --tag "$TAG")

    FI=$(get fi)
    if [[ -n "$FI" ]]; then
        read -r FI_LEVEL FI_STEP FI_TARGET FI_BIT <<< "$FI"
        APP_ARGS+=(--fi "$FI_LEVEL" "$FI_STEP" "$FI_TARGET" "$FI_BIT")
        FI_RANK=$(get fi_rank); [[ -n "$FI_RANK" ]] && APP_ARGS+=(--fi-rank "$FI_RANK")
    fi

    V=$(get snap);    [[ -n "$V" ]] && APP_ARGS+=(--snap "$V")
    V=$(get kernel);  [[ -n "$V" ]] && APP_ARGS+=(--kernel "$V")
    V=$(get scatter); [[ -n "$V" ]] && APP_ARGS+=(--scatter "$V")
    V=$(get thr);     [[ -n "$V" ]] && APP_ARGS+=(--thr "$V")
    V=$(get ic);      [[ -n "$V" ]] && APP_ARGS+=(--ic "$V")
    V=$(get diag_cg); [[ -n "$V" ]] && APP_ARGS+=(--diag-cg "$V")

    [[ "$(get reproject-bc)" == "true" ]] && APP_ARGS+=(--reproject-bc)
    [[ "$(get detect)"       == "true" ]] && APP_ARGS+=(--detect)
    [[ "$(get verify)"       == "true" ]] && APP_ARGS+=(--verify)
    [[ "$(get no_source)"    == "true" ]] && APP_ARGS+=(--no-source)
    [[ "$(get dump_snaps)"   == "true" ]] && APP_ARGS+=(--dump-snaps)

    QBLOCK=$(get qblock)
    if [[ -n "$QBLOCK" ]]; then
        read -r QBLOCK_X QBLOCK_Y QBLOCK_Z QBLOCK_M <<< "$QBLOCK"
        APP_ARGS+=(--qblock "$QBLOCK_X" "$QBLOCK_Y" "$QBLOCK_Z" "$QBLOCK_M")
    fi

    echo "Executing configuration: $TAG"

    printf 'nsys profile '
    printf '%q ' "${NSYS_ARGS[@]}"
    printf 'mpirun -np %s ./build/heat_solver_het ' "$NP"
    printf '%q ' "${APP_ARGS[@]}"
    printf '\n'

    nsys profile \
            "${NSYS_ARGS[@]}" \
            mpirun -np "$NP" \
            ./build/heat_solver_het \
            "${APP_ARGS[@]}"

done 3< <(jq -c '.configurations[]' "$CONFIG_FILE")
