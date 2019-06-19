node('gpu') {
    timestamps {
        try {
            stage('Clean') {
                sh "rm -rf .[^.] .??* *"
            }
            stage('Checkout') {
                sh "cp -r ${pwd()}@script/* ."
            }
            stage('Setup') {
                env.TFHUB_CACHE_DIR="tfhub_cache"
                env.LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64"
                sh """
                    virtualenv --python=python3 '.venv-$BUILD_NUMBER'
                    . '.venv-$BUILD_NUMBER/bin/activate'
                    pip install .[tests,docs]
                    pip install -r deeppavlov/requirements/tf-gpu.txt
                    rm -rf `find . -mindepth 1 -maxdepth 1 ! -name tests ! -name Jenkinsfile ! -name docs ! -name '.venv-$BUILD_NUMBER'`
                """
            }
            stage('Tests') {
                sh """
                    . .venv-$BUILD_NUMBER/bin/activate
                    pytest -v --disable-warnings
                    cd docs
                    make clean
                    make html
                """
                currentBuild.result = 'SUCCESS'
            }
        }
        catch(e) {
            currentBuild.result = 'FAILURE'
            throw e
        }
        finally {
            emailext to: '${DEFAULT_RECIPIENTS}',
                subject: "${env.JOB_NAME} - Build # ${currentBuild.number} - ${currentBuild.result}!",
                body: '${BRANCH_NAME} - ${BUILD_URL}',
                attachLog: true
        }
    }
}
