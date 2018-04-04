node('dgx') {
    try {
        /*stage('Clean') {
            sh """
                rm -rf .[^.] .??* *
            """
        }*/
        stage('Checkout') {
            git branch: 'dev', url: 'https://github.com/deepmipt/deeppavlov.git'
        }
        stage('Setup') {
            env.CUDA_VISIBLE_DEVICES=0
            sh """
                virtualenv --python=python3 ".venv-$BUILD_NUMBER"
                . .venv-$BUILD_NUMBER/bin/activate
                pip install pip==9.0.3
                python setup.py develop
                python -m spacy download en
                pip3 install -r requirements-dev.txt
            """
        }
        stage('Tests') {
            sh """
                . .venv-$BUILD_NUMBER/bin/activate
                cd tests
                pytest -v
            """
        }
    } catch (e) {
        emailext attachLog: true, subject: '$PROJECT_NAME - Build # $BUILD_NUMBER - FAILED!'
        throw e
    }
    emailext attachLog: true, subject: '$PROJECT_NAME - Build # $BUILD_NUMBER - $BUILD_STATUS!'
}