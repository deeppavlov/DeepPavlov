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
                virtualenv --python=python3 "tests/.venv-$BUILD_NUMBER"
                . tests/.venv-$BUILD_NUMBER/bin/activate
                pip install -r dp_requirements/tf-gpu.txt
                pip install .[tests]
                rm `find . -mindepth 1 -maxdepth 1 ! -name tests ! -name Jenkinsfile`
            """
        }
        stage('Tests') {
            sh """
                . tests/.venv-$BUILD_NUMBER/bin/activate
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
