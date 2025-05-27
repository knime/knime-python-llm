/*
 * ------------------------------------------------------------------------
 *
 *  Copyright by KNIME AG, Zurich, Switzerland
 *  Website: http://www.knime.com; Email: contact@knime.com
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License, Version 3, as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, see <http://www.gnu.org/licenses>.
 *
 *  Additional permission under GNU GPL version 3 section 7:
 *
 *  KNIME interoperates with ECLIPSE solely via ECLIPSE's plug-in APIs.
 *  Hence, KNIME and ECLIPSE are both independent programs and are not
 *  derived from each other. Should, however, the interpretation of the
 *  GNU GPL Version 3 ("License") under any applicable laws result in
 *  KNIME and ECLIPSE being a combined program, KNIME AG herewith grants
 *  you the additional permission to use and propagate KNIME together with
 *  ECLIPSE with only the license terms in place for ECLIPSE applying to
 *  ECLIPSE and the GNU GPL Version 3 applying for KNIME, provided the
 *  license terms of ECLIPSE themselves allow for the respective use and
 *  propagation of ECLIPSE together with KNIME.
 *
 *  Additional permission relating to nodes for KNIME that extend the Node
 *  Extension (and in particular that are based on subclasses of NodeModel,
 *  NodeDialog, and NodeView) and that only interoperate with KNIME through
 *  standard APIs ("Nodes"):
 *  Nodes are deemed to be separate and independent programs and to not be
 *  covered works.  Notwithstanding anything to the contrary in the
 *  License, the License does not apply to Nodes, you are not required to
 *  license Nodes under the License, and you are granted a license to
 *  prepare and propagate Nodes, in each case even if such Nodes are
 *  propagated with or for interoperation with KNIME.  The owner of a Node
 *  may freely choose the license terms applicable to such Node, including
 *  when such Node is propagated with or for interoperation with KNIME.
 * ---------------------------------------------------------------------
 *
 * History
 *   9 Dec 2019 (Marc Bux, KNIME GmbH, Berlin, Germany): created
 */
package org.knime.python.llm.java.writer;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.DirectoryNotEmptyException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.lang3.ObjectUtils;
import org.knime.python.llm.java.ExistsOption;
import org.knime.python.llm.java.manipulate.WorkflowSegmentManipulations;
import org.knime.core.node.BufferedDataTable;
import org.knime.core.node.CanceledExecutionException;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.context.NodeCreationConfiguration;
import org.knime.core.node.port.PortObject;
import org.knime.core.node.port.PortObjectSpec;
import org.knime.core.node.port.PortType;
import org.knime.core.node.util.CheckUtils;
import org.knime.core.node.util.ConvenienceMethods;
import org.knime.core.node.workflow.NodeUIInformation;
import org.knime.core.node.workflow.VariableTypeRegistry;
import org.knime.core.node.workflow.WorkflowManager;
import org.knime.core.node.workflow.capture.BuildWorkflowsUtil;
import org.knime.core.node.workflow.capture.ReferenceReaderDataUtil;
import org.knime.core.node.workflow.capture.WorkflowPortObject;
import org.knime.core.node.workflow.capture.WorkflowPortObjectSpec;
import org.knime.core.node.workflow.capture.WorkflowSegment;
import org.knime.core.node.workflow.capture.WorkflowSegment.IOInfo;
import org.knime.core.node.workflow.capture.WorkflowSegment.Input;
import org.knime.core.node.workflow.capture.WorkflowSegment.Output;
import org.knime.core.node.workflow.capture.WorkflowSegment.PortID;
import org.knime.core.util.FileUtil;
import org.knime.core.util.LockFailedException;
import org.knime.core.util.Pair;
import org.knime.core.util.UniqueNameGenerator;
import org.knime.core.util.VMFileLocker;
import org.knime.filehandling.core.connections.FSFiles;
import org.knime.filehandling.core.connections.FSPath;
import org.knime.filehandling.core.connections.base.UnixStylePathUtil;
import org.knime.filehandling.core.data.location.variable.FSLocationVariableType;
import org.knime.filehandling.core.node.portobject.writer.PortObjectToPathWriterNodeModel;

/**
 * Workflow writer node.
 *
 * @author Marc Bux, KNIME GmbH, Berlin, Germany
 * @author Martin Horn, KNIME GmbH, Konstanz, Germany
 */
public final class WorkflowWriterNodeModel extends PortObjectToPathWriterNodeModel<WorkflowWriterNodeConfig> {

    private final boolean m_useV2SmartInOutNames;

    WorkflowWriterNodeModel(final NodeCreationConfiguration creationConfig, final boolean useV2SmartInOutNames) {
        super(creationConfig, new WorkflowWriterNodeConfig(creationConfig));
        m_useV2SmartInOutNames = useV2SmartInOutNames;
    }

    @Override
    protected void configureInternal(final PortObjectSpec[] inSpecs) throws InvalidSettingsException {

        final WorkflowWriterNodeConfig config = getConfig();
        final var workflowPortObjectSpec =
            validateAndGetWorkflowPortObjectSpec(inSpecs[getInputTableIndex()], InvalidSettingsException::new);

        final Optional<String> err = validateWorkflowName(workflowPortObjectSpec,
            config.isUseCustomName().getBooleanValue(), config.getCustomName().getStringValue());
        if (err.isPresent()) {
            throw new InvalidSettingsException(err.get());
        }

        config.getIONodes().validateSettings();
    }

    @SuppressWarnings("resource")
    @Override
    protected void writeToPath(final PortObject object, final Path outputPath, final ExecutionContext exec)
        throws Exception {

        CheckUtils.checkArgumentNotNull(object, "WorkflowPortObject must not be null.");
        CheckUtils.checkArgumentNotNull(outputPath, "Output Path must not be null.");
        CheckUtils.checkArgumentNotNull(exec, "Execution Context must not be null.");

        final WorkflowWriterNodeConfig config = getConfig();

        final var workflowPortObject = (WorkflowPortObject)object;
        final var workflowPortObjectSpec = workflowPortObject.getSpec();
        final var segment = workflowPortObjectSpec.getWorkflowSegment();
        final var archive = config.isArchive().getBooleanValue();
        final var openAfterWrite = config.isOpenAfterWrite().getBooleanValue();
        final boolean overwrite =
            config.getExistsOption().getStringValue().equals(ExistsOption.OVERWRITE.getActionCommand());

        if (config.getDoUpdateTemplateLinks().getBooleanValue()) {
            WorkflowSegmentManipulations.UPDATE_LINKED_TEMPLATES.apply(segment);
        }
        if (config.getDoRemoveTemplateLinks().getBooleanValue()) {
            WorkflowSegmentManipulations.REMOVE_TEMPLATE_LINKS.apply(segment);
        }

        // determine workflow name
        final String workflowName;
        if (config.isUseCustomName().getBooleanValue()) {
            workflowName = config.getCustomName().getStringValue();
        } else {
            final String originalName = workflowPortObjectSpec.getWorkflowName();
            if (originalName == null || originalName.isEmpty()) {
                throw new InvalidSettingsException(
                    "Default workflow name is null or empty. Consider using a custom workflow name.");
            }
            workflowName = determineWorkflowName(workflowPortObjectSpec);
            // Custom workflow names provided by upstream nodes are assumed to be correct, so no warning
            // needs to be given here. The name is still escaped of any invalid characters.
        }

        // create directory at output path, if applicable (parent path was already checked in super class)
        if (!Files.exists(outputPath)) {
            FSFiles.createDirectories(outputPath);
        }

        // resolve destination path and check if it is present already
        final FSPath dest;
        if (archive) {
            dest = (FSPath) outputPath.resolve(String.format("%s.knwf", workflowName));
        } else {
            dest = (FSPath) outputPath.resolve(workflowName);
        }
        if (Files.exists(dest)) {
            if (!overwrite) {
                throw new InvalidSettingsException(String
                    .format("Destination path \"%s\" exists and must not be overwritten due to user settings.", dest));
            }
            if (!archive && Files.exists(dest.resolve(VMFileLocker.LOCK_FILE))) {
                throw new InvalidSettingsException(String.format("To-be-overwritten workflow \"%s\" is locked.", dest));
            }
        }

        // create temporary local directory
        exec.setProgress(.33, () -> "Saving workflow to disk.");
        final var tmpDir = FileUtil.createTempDir("workflow-writer");
        final File localSource = write(tmpDir, workflowName, segment, exec, config.getIONodes(),
            m_useV2SmartInOutNames, workflowPortObject, archive, this::setWarningMessage);

        final var localSourcePath = localSource.toPath();

        // copy workflow from temporary source to desired destination
        exec.setProgress(.67, () -> "Copying workflow to destination.");
        final var workflowAware = dest.getFileSystem().getWorkflowAware();
        if (archive) {
            if (overwrite && Files.exists(dest)) {
                Files.copy(localSourcePath, dest, StandardCopyOption.REPLACE_EXISTING);
            } else {
                Files.copy(localSourcePath, dest);
            }
        } else if (workflowAware.isPresent()) {
            workflowAware.orElseThrow().deployWorkflow(localSourcePath, dest, overwrite, openAfterWrite);
        } else {
            try (final Stream<Path> streams = Files.walk(localSourcePath)) {
                for (final Path path : streams.collect(Collectors.toList())) {
                    final var rel = localSourcePath.relativize(path);
                    final var relString = UnixStylePathUtil.asUnixStylePath(rel.toString());
                    final Path res = dest.resolve(relString);
                    exec.setMessage(() -> String.format("Copying file %s.", relString));
                    if (overwrite) {
                        try {
                            Files.copy(path, res, StandardCopyOption.REPLACE_EXISTING);
                        } catch (DirectoryNotEmptyException e) {
                            // we do not care about these when in overwrite mode
                        }
                    } else {
                        Files.copy(path, res);
                    }
                }
            }
        }
        UniqueNameGenerator nameGen = new UniqueNameGenerator(
            getAvailableFlowVariables(VariableTypeRegistry.getInstance().getAllTypes()).keySet());
        pushFlowVariable(nameGen.newName("workflow-location"), FSLocationVariableType.INSTANCE, dest.toFSLocation());
        FileUtil.deleteRecursively(tmpDir);
    }

    /**
     * @param tmpDir
     * @param workflowName
     * @param segment
     * @param exec
     * @param ioNodes
     * @param useV2SmartInOutNames
     * @param workflowPortObject
     * @param archive
     * @param warningMessageConsumer
     * @return
     * @throws Exception
     */
    public static File write(final File tmpDir, final String workflowName, final WorkflowSegment segment,
        final ExecutionContext exec, final SettingsModelIONodes ioNodes, final boolean useV2SmartInOutNames,
        final WorkflowPortObject workflowPortObject, final boolean archive,
        final Consumer<String> warningMessageConsumer) throws Exception {

        final WorkflowManager wfm = BuildWorkflowsUtil.loadWorkflow(segment, warningMessageConsumer);
        wfm.setName(workflowName);
        addIONodes(wfm, ioNodes, useV2SmartInOutNames, workflowPortObject, exec, warningMessageConsumer);

        return writeWorkflowPortObjectAndReferencedData(wfm, tmpDir, segment, exec, archive);
    }

    private static File writeWorkflowPortObjectAndReferencedData(final WorkflowManager wfm, final File tmpDir,
        final WorkflowSegment segment, final ExecutionContext exec, final boolean archive) throws IOException,
        CanceledExecutionException, URISyntaxException, InvalidSettingsException, LockFailedException {
        final var tmpWorkflowDir = new File(tmpDir, wfm.getName());
        tmpWorkflowDir.mkdir();
        final var tmpDataDir = new File(tmpWorkflowDir, "data");
        tmpDataDir.mkdir();

        try {
            ReferenceReaderDataUtil.writeReferenceReaderData(wfm, segment.getPortObjectReferenceReaderNodes(),
                tmpDataDir, exec);
            wfm.save(tmpWorkflowDir, exec.createSubProgress(.34), false);
        } finally {
            segment.disposeWorkflow();
        }

        // zip temporary directory if applicable
        if (!archive) {
            return tmpWorkflowDir;
        }
        final var localSource = new File(tmpDir, String.format("%s.knwf", wfm.getName()));
        FileUtil.zipDir(localSource, tmpWorkflowDir, 9);
        return localSource;
    }


    static void validateIONodeConfigs(final Map<String, ? extends IOInfo> ioSpecs,
        final Map<String, ? extends IONodeConfig> ioNodeConfigs) throws InvalidSettingsException {
        // inputs and outputs that are present in the workflow fragment -- expecting all to be configured
        var validNamesSet = new HashSet<>(ioSpecs.keySet());

        for (Entry<String, ? extends IONodeConfig> nodeConfig : ioNodeConfigs.entrySet()) {
            var parameterName = nodeConfig.getKey();
            var ioInfo = ioSpecs.get(parameterName);
            CheckUtils.checkSettingNotNull(ioInfo,
                "Node was configured for \"%s\" but no such parameter exists in the workflow fragment",
                parameterName);
            validNamesSet.remove(parameterName);

            var type = ioInfo.getType().orElseThrow(() -> new InvalidSettingsException(
                    "Workflow fragment contains port types not present in this installation (plug-in installed?)"));
            validateIONodeConfig(type, nodeConfig.getValue());
        }


        CheckUtils.checkSetting(validNamesSet.isEmpty(),
            "Found nodes in the workflow fragment that are not configured (%s)",
            ConvenienceMethods.getShortStringFrom(validNamesSet, 3));
    }


    private static void validateIONodeConfig(final PortType portType, final IONodeConfig config)
        throws InvalidSettingsException {
        if (ObjectUtils.isEmpty(config)) { // NONE_CHOICE was selected, don't validate it
            return;
        }
        var isDataTableSupported = true;
        if (config instanceof DataTableConfigurator) {
            isDataTableSupported = portType.equals(BufferedDataTable.TYPE);
        }
        CheckUtils.checkSetting(isDataTableSupported, "The %s supports only Data Table port type", config.getNodeName());
    }

    private static void addIONodes(final WorkflowManager wfm, final SettingsModelIONodes ioNodes,
        final boolean useV2SmartInOutNames, final WorkflowPortObject workflowPortObject, final ExecutionContext exec,
        final Consumer<String> warningMessageConsumer) throws InvalidSettingsException {
        exec.setMessage(() -> "Adding input and output nodes");

        ioNodes
            .initWithDefaults(workflowPortObject.getSpec().getInputIDs(),
            workflowPortObject.getSpec().getOutputIDs());

        int[] wfmb = NodeUIInformation.getBoundingBoxOf(wfm.getNodeContainers());

        List<String> configuredInputs = ioNodes.getConfiguredInputs(workflowPortObject.getSpec().getInputIDs());
        List<String> configuredOutputs = ioNodes.getConfiguredOutputs(workflowPortObject.getSpec().getOutputIDs());

        Map<String, Input> inputs = workflowPortObject.getSpec().getInputs();
        Map<String, Output> outputs = workflowPortObject.getSpec().getOutputs();

        Pair<Integer, int[]> inputPositions = BuildWorkflowsUtil.getInputOutputNodePositions(wfmb, inputs.size(), true);
        Pair<Integer, int[]> outputPositions = BuildWorkflowsUtil.getInputOutputNodePositions(wfmb, outputs.size(), false);

        // add, connect and configure inputs
        var i = 0;
        for (String id : configuredInputs) {
            // add and connect
            IONodeConfig inConfig = ioNodes.getInputNodeConfig(id).get();
            var portType = inputs.get(id).getType().orElse(null);
            var x = inputPositions.getFirst();
            var y = inputPositions.getSecond()[i];
            var ports = inputs.get(id).getConnectedPorts().stream();
            var nodeID = inConfig.addAndConnectIONode(wfm, portType, ports, x, y);

            // configure (optional data table for specific implementations)
            var dataTable = workflowPortObject.getInputDataFor(id).orElse(null);
            inConfig.configureIONode(wfm, nodeID, useV2SmartInOutNames, dataTable);
            i++;
        }

        // add, connect and configure outputs
        i = 0;
        for (String id : configuredOutputs) {
            // add and connect
            IONodeConfig outConfig = ioNodes.getOutputNodeConfig(id).get();
            var portType = outputs.get(id).getType().orElse(null);
            var x = outputPositions.getFirst();
            var y = outputPositions.getSecond()[i];
            var ports = getOutputPortIdStream(outputs, id);
            var nodeID = outConfig.addAndConnectIONode(wfm, portType, ports, x, y);

            // configure
            outConfig.configureIONode(wfm, nodeID, useV2SmartInOutNames, null);
            i++;
        }

        boolean unconnectedInputs = inputs.size() > configuredInputs.size();
        boolean unconnectedOutputs = outputs.size() > configuredOutputs.size();
        if (unconnectedInputs || unconnectedOutputs) {
            warningMessageConsumer.accept(
                "Some " + (unconnectedInputs ? "input" : "") + (unconnectedInputs && unconnectedOutputs ? " and " : "")
                    + (unconnectedOutputs ? "output" : "") + " ports are not connected.");
        }
    }

    /**
     * @param outputs a map of id to {@link Output}
     * @param id the id of the Output Node
     * @return a stream of {@link PortID}
     */
    private static Stream<PortID> getOutputPortIdStream(final Map<String, Output> outputs, final String id) {
        Optional<PortID> connectedPort = outputs.get(id).getConnectedPort();
        return connectedPort.isPresent() ? Stream.of(connectedPort.get()) : Stream.empty();
    }


    public static Optional<String> validateWorkflowName(final WorkflowPortObjectSpec portObjectSpec,
        final boolean useCustomName, final String customName) {
        if (useCustomName) {
            return BuildWorkflowsUtil.validateCustomWorkflowName(customName, false, false);
        } else if (determineWorkflowName(portObjectSpec).isEmpty()) {
            return Optional.of("Default workflow name is empty. Consider using a custom workflow name.");
        }
        return Optional.empty();
    }

    public static final String determineWorkflowName(final WorkflowPortObjectSpec spec) {
        return FileUtil.ILLEGAL_FILENAME_CHARS_PATTERN.matcher(spec.getWorkflowName()).replaceAll("_").trim();
    }

    public static <E extends Exception> WorkflowPortObjectSpec validateAndGetWorkflowPortObjectSpec(
        final PortObjectSpec spec, final Function<String, E> errorFunction) throws E {

        final WorkflowPortObjectSpec portObjectSpec = (WorkflowPortObjectSpec)spec;
        if (portObjectSpec == null) {
            throw errorFunction.apply("No workflow available.");
        }
        return portObjectSpec;
    }

}
