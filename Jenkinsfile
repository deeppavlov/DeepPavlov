node('gpu') {
    try {
        stage('Checkout') {
            git branch: '${GITHUB_PR_SOURCE_BRANCH}', url: 'https://github.com/deepmipt/deeppavlov.git'
            if (${GITHUB_PR_NUMBER} > 0) {
                sh """
                    echo 'I am here!'
                    git checkout ${GITHUB_PR_TARGET_BRANCH}
                    git merge ${GITHUB_PR_SOURCE_BRANCH}
                """
            }
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
                pytest -v -m "ner"
            """
        }
    } catch (e) {
        emailext to: '${DEFAULT_RECIPIENTS},${GITHUB_PR_AUTHOR_EMAIL}',
            subject: '$PROJECT_NAME - Build # $BUILD_NUMBER - FAILED!',
            body: '${BRANCH_NAME} - ${BUILD_URL}',
            attachLog: true
        throw e
    }
    emailext to: '${DEFAULT_RECIPIENTS},${GITHUB_PR_AUTHOR_EMAIL}',
        subject: '$PROJECT_NAME - Build # $BUILD_NUMBER - $BUILD_STATUS!',
        body: '${BRANCH_NAME} - ${BUILD_URL}',
        attachLog: true
}
