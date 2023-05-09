#!/bin/bash
SEEDS=(24921 34321 82931 56122 73192 10932 69347 48369 99918 66182)
EXP_SEEDS=(9288012 4527271 3729391 1272922 5362782 2893817 7828172 1093802 8199829 6545288)
TASKS_CNT=(4 7 14)

for K in {1..4};
    do
        for tks in {1..3}
            do
                for i in {1..10}
                    do
                        for j in {1..10}
                        do
                            ( SEED=${SEEDS[$i-1]}
                            EXP_SEED=${EXP_SEEDS[$j-1]}
                            TASKS=${TASKS_CNT[$tks-1]}
                            echo "Seed: $SEED, ExplorationSeed: $EXP_SEED"
                            echo "K: $K, Tasks: $TASKS"
                            python robotarium_main_bulk.py --k $K --task $TASKS --seed $SEED --exp_seed $EXP_SEED > logs/out_${SEED}_TS${TASKS}_K${K}_EXP${EXP_SEED}_base_false.log ) &
                            #python hello.py
                        done
                    done
            done
    done