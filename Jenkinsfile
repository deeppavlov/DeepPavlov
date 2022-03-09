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
                    EPOCH=$(date +%s) docker-compose -f utils/Docker/docker-compose.yml -p $BUILD_TAG build
                """
            }
            stage('Tests') {
                sh """
                    docker-compose -f utils/Docker/docker-compose.yml -p $BUILD_TAG up py36 py37
                    docker-compose -f utils/Docker/docker-compose.yml -p $BUILD_TAG up py38 py39
                    docker-compose -f utils/Docker/docker-compose.yml -p $BUILD_TAG rm -f
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
