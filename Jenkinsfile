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
                    EPOCH=\$(date +%s) docker-compose -f utils/Docker/docker-compose.yml -p $BUILD_TAG build
                """
            }
            stage('Tests') {
                sh """
                    docker-compose -f utils/Docker/docker-compose.yml -p $BUILD_TAG up py36 py37
                    docker-compose -f utils/Docker/docker-compose.yml -p $BUILD_TAG ps | grep Exit | grep -v 'Exit 0' && exit 1
                    docker-compose -f utils/Docker/docker-compose.yml -p $BUILD_TAG up py38 py39
                    docker-compose -f utils/Docker/docker-compose.yml -p $BUILD_TAG ps | grep Exit | grep -v 'Exit 0' && exit 1
                    docker-compose -f utils/Docker/docker-compose.yml -p $BUILD_TAG up py310 py311
                    docker-compose -f utils/Docker/docker-compose.yml -p $BUILD_TAG ps | grep Exit | grep -v 'Exit 0' && exit 1 || exit 0
                """
                currentBuild.result = 'SUCCESS'
            }
        }
        catch(e) {
            currentBuild.result = 'FAILURE'
            throw e
        }
        finally {
            sh """
                docker-compose -f utils/Docker/docker-compose.yml -p $BUILD_TAG rm -f
                docker network rm \$(echo $BUILD_TAG | awk '{print tolower(\$0)}')_default
            """
            emailext to: "\${DEFAULT_RECIPIENTS}",
                subject: "${env.JOB_NAME} - Build # ${currentBuild.number} - ${currentBuild.result}!",
                body: '${BRANCH_NAME} - ${BUILD_URL}',
                attachLog: true
        }
    }
}
