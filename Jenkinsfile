#!groovy
def BN = (BRANCH_NAME == 'master' || BRANCH_NAME.startsWith('releases/')) ? BRANCH_NAME : 'releases/2023-12'

// The knime version defines which parent pom is used in the built feature. The version was added in KNIME 4.7.0 and
// will have to be updated with later KNIME versions.
def knimeVersion = '5.3'

def repositoryName = "knime-python-llm"
def featureName = 'org.knime.python.features.llm.feature.group'
def extensionPath = "." // Path to knime.yml file


def targetDirectory = "${repositoryName}-build"

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

                def outputPath = "output"

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
                        micromamba run -p ${prefixPath} python knime-bundling/scripts/build_python_extension.py ${extensionPath} ${outputPath} --force --knime-version ${knimeVersion} --knime_build
                        """
                    }
                   // todo:  when knime-python-bundling is merged replace with this, as we can just use the conda build then
                   // micromamba run -p ${prefixPath} build_python_extension.py ${extensionPath} ${outputPath} -f --knime-version ${knimeVersion} --knime_build

                }
            }
            stage("Deploy p2") {
                    env.lastStage = env.STAGE_NAME
                    p2Tools.deploy("update")
                    println("Deployed")

                }
            runWorkflowTests(
                stageName: "${targetDirectory}: Online Install",
                testflowsPath: repositoryName,
                feature: featureName,
                branchName: BN,
                repositories: [
                        'knime-python',
                        'knime-core-columnar',
                        'knime-testing-internal',
                        'knime-python-legacy',
                        'knime-conda',
                        'knime-python-bundling',
                        'knime-credentials-base',
                        'knime-gateway',
                        repositoryName
                        ]
           )
           if (BN == 'master' && currentBuild.result == 'STABLE'){ // Do we need to build before the test?
                stage("Deploy p2") {
                    env.lastStage = env.STAGE_NAME
                    p2Tools.deploy("update")
                    println("Deployed")

                }
            }

        }
    }
 } catch (ex) {
    currentBuild.result = 'FAILURE'
    throw ex
 } finally {
    notifications.notifyBuild(currentBuild.result)
}


/*
 * Runs the workflow tests for the given feature.
 *
 * @param stageName Name of the stage in the Jenkins pipeline.
 * @param feature Name of the feature.
 * @param branchName Name of the branch.
 * @param repositories List of repositories to install for the tests.
 * @param testflowsPath subdirectory of the testflows directory to run the tests from. Usually the name of the repositories
    but for instance in knime-python-bundling it is "knime-python-bundling/testext"
*/
void runWorkflowTests(Map args = [:]) {
    timeout(time: 30, unit: 'MINUTES') {
        stage(args.stageName) {
            def testBaseBranch = args.branchName == KNIMEConstants.NEXT_RELEASE_BRANCH ? "master" : args.branchName.replace("releases/","")
            workflowTests.runTests(
                testflowsDir: "Testflows (${testBaseBranch})/${args.testflowsPath}",
                configurations: workflowTests.DEFAULT_CONFIGURATIONS,
                dependencies: [
                    repositories: args.repositories,
                    ius: [args.feature]
                ],
            )
        }
    }
}