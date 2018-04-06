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
                pip install pip==9.0.3
                python setup.py develop
                python -m spacy download en
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