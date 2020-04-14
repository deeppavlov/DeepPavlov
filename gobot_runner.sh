DEEPPAVLOV_DIR=$(pwd)
GOBOT_CONFIG_FILES=$DEEPPAVLOV_DIR/deeppavlov/configs/go_bot/*.json
PY_TEST_RUNNER=gobot_explore/gobot_runner.py
DOWNLOAD_PARAM_CANDIDATES=(True False)

export PYTHONPATH=$DEEPPAVLOV_DIR:$PYTHONPATH

for config_fn in $GOBOT_CONFIG_FILES; do
    for dld_param in ${DOWNLOAD_PARAM_CANDIDATES[@]}; do
        echo $(date) BEFORE $PY_TEST_RUNNER $config_fn $dld_param
        
        python3 $PY_TEST_RUNNER $config_fn $dld_param \
         >> ${config_fn}_${dld_param}.log \
	 2>> ${config_fn}_${dld_param}.err
        
        rm -r gobot_runner_dir
        rm -r ~/.deeppavlov/models/go*

        echo $(date) AFTER $PY_TEST_RUNNER $config_fn $dld_param 


    done
done;
