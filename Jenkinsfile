node('cuda-module') {
    timestamps {
        try {
            stage('Clean') {
                sh "rm -rf .[^.] .??* *"
            }
            stage('Checkout') {
                checkout scm
            }
            stage('Setup') {
                env.TFHUB_CACHE_DIR="tfhub_cache"
                sh """
                    virtualenv --python=python3.7 '.venv-$BUILD_NUMBER'
                    . '.venv-$BUILD_NUMBER/bin/activate'
                    pip install .[tests,docs]
                    pip install -r deeppavlov/requirements/tf-gpu.txt
                    rm -rf `find . -mindepth 1 -maxdepth 1 ! -name tests ! -name Jenkinsfile ! -name docs ! -name '.venv-$BUILD_NUMBER'`
                """
            }
            stage('Tests') {
                sh """
                    . /etc/profile
                    module add cuda/10.0
                    . .venv-$BUILD_NUMBER/bin/activate

                    cd docs
                    make clean
                    make html
                    cd ..

                    flake8 `python -c 'import deeppavlov; print(deeppavlov.__path__[0])'` --count --select=E9,F63,F7,F82 --show-source --statistics
                    pytest -v --disable-warnings
                """
                currentBuild.result = 'SUCCESS'
            }
        }
        catch(e) {
            currentBuild.result = 'FAILURE'
            throw e
        }
        finally {
            emailext to: "\${DEFAULT_RECIPIENTS}",
                subject: "${env.JOB_NAME} - Build # ${currentBuild.number} - ${currentBuild.result}!",
                body: '${BRANCH_NAME} - ${BUILD_URL}',
                attachLog: true
        }
    }
}
