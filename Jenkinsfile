node('gpu') {
    try {
        stage('Clean') {
            sh "rm -rf .[^.] .??* *"
        }
        stage('Checkout') {
            sh "cp -r ${pwd()}@script/* ."
        }
        stage('Setup') {
            env.CUDA_VISIBLE_DEVICES=0
            sh """
                virtualenv --python=python3 ".venv-$BUILD_NUMBER"
                . .venv-$BUILD_NUMBER/bin/activate
                sed -ri 's/^ *tensorflow *(=|<|>|\$)/tensorflow-gpu\\1/g' requirements.txt
                sed -i "s/stream=True/stream=False/g" deeppavlov/core/data/utils.py
                python setup.py develop
                pip install http://lnsigo.mipt.ru/export/en_core_web_sm-2.0.0.tar.gz
                python -m spacy link en_core_web_sm en --force
                pip install -r requirements-dev.txt
            """
        }
        stage('Tests') {
            sh """
                . .venv-$BUILD_NUMBER/bin/activate
                pytest -v
            """
        }
    } catch (e) {
        emailext to: '${DEFAULT_RECIPIENTS}',
            subject: '${PROJECT_NAME} - Build # ${BUILD_NUMBER} - FAILED!',
            body: '${BRANCH_NAME} - ${BUILD_URL}',
            attachLog: true
        throw e
    }
    emailext to: '${DEFAULT_RECIPIENTS}',
        subject: '${PROJECT_NAME} - Build # ${BUILD_NUMBER} - ${BUILD_STATUS}!',
        body: '${BRANCH_NAME} - ${BUILD_URL}',
        attachLog: true
}
