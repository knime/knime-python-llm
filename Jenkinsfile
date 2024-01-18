#!groovy
def BN = (BRANCH_NAME == 'master' || BRANCH_NAME.startsWith('releases/')) ? BRANCH_NAME : 'releases/2023-12'

// The knime version defines which parent pom is used in the built feature. The version was added in KNIME 4.7.0 and
// will have to be updated with later KNIME versions.
def knimeVersion = '5.3'

def repositoryName = "knime-python-llm"

def extensionPath = "." // Path to knime.yml file
def outputPath = "output"

library "knime-pipeline@$BN"

properties([
    parameters([p2Tools.getP2pruningParameter()]),
    buildDiscarder(logRotator(numToKeepStr: '5')),
    disableConcurrentBuilds()
])

try {
    timeout(time: 100, unit: 'MINUTES') {
        node('workflow-tests && ubuntu22.04 && java17') {
            stage("Checkout Sources") {
                env.lastStage = env.STAGE_NAME
                checkout scm
            }

            stage("Create Conda Env"){
                env.lastStage = env.STAGE_NAME
                prefixPath = "${WORKSPACE}/${repositoryName}"
                condaHelpers.createCondaEnv(prefixPath: prefixPath, pythonVersion:'3.9', packageNames: ['knime-extension-bundling'])
            }
            stage("Build Python Extension") {
                env.lastStage = env.STAGE_NAME

                // todo:  when knime-python-bundling is merged remove this, as we can just use the conda build then
                // Checkout the knime-python-bundling repository
                checkout([
                    $class: 'GitSCM',
                    branches: [[name: 'remotes/origin/AP-20676-move-knime-built-python-extensions-to-internal-update-site2' ]],
                    extensions: [[$class: 'RelativeTargetDirectory', relativeTargetDir: 'knime-bundling'], [$class: 'GitLFSPull']],
                    userRemoteConfigs: [[ credentialsId: 'bitbucket-jenkins', url: 'https://bitbucket.org/KNIME/knime-python-bundling' ]]
                ])

                withEnv([ "MVN_OPTIONS=-Dknime.p2.repo=https://jenkins.devops.knime.com/p2/knime/" ]) {
                    withCredentials([usernamePassword(credentialsId: 'ARTIFACTORY_CREDENTIALS', passwordVariable: 'ARTIFACTORY_PASSWORD', usernameVariable: 'ARTIFACTORY_LOGIN'),
                    ]) {
                        sh """
                        micromamba run -p ${prefixPath} python knime-bundling/scripts/build_python_extension.py ${extensionPath} ${outputPath} --force --knime-version ${knimeVersion} --knime_build --excluded-files ${prefixPath} knime-bundling knime-bundling@tmp
                        """
                    }
                   // todo:  when knime-python-bundling is merged replace with this, as we can just use the conda build then
                   // micromamba run -p ${prefixPath} build_python_extension.py ${extensionPath} ${outputPath} -f --knime-version ${knimeVersion} --knime_build --excluded-files ${repositoryName}

                }
            }
            stage("Deploy p2") {
                    env.lastStage = env.STAGE_NAME
                    p2Tools.deploy(outputPath)
                    println("Deployed")

                }
            workflowTests.runTests(
                configurations: workflowTests.DEFAULT_CONFIGURATIONS,
                dependencies: [
                    repositories: [
                        'knime-python',
                        'knime-python-types',
                        'knime-core-columnar',
                        'knime-testing-internal',
                        'knime-python-legacy',
                        'knime-conda',
                        'knime-python-bundling',
                        'knime-credentials-base',
                        'knime-gateway',
                        'knime-base',
                        repositoryName
                        ],
                ],
            )
        }
    }
} catch (ex) {
    currentBuild.result = 'FAILURE'
    throw ex
} finally {
    notifications.notifyBuild(currentBuild.result)
}
