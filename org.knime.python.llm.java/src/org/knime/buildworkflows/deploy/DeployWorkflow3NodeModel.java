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
 *   Nov 22, 2020 (Mark Ortmann, KNIME GmbH, Berlin, Germany): created
 */
package org.knime.python.llm.java.deploy;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.EnumSet;

import org.apache.commons.lang3.StringUtils;
import org.knime.python.llm.java.ExistsOption;
import org.knime.python.llm.java.manipulate.WorkflowSegmentManipulations;
import org.knime.python.llm.java.writer.WorkflowWriterNodeModel;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.ExecutionMonitor;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeModel;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.context.ports.PortsConfiguration;
import org.knime.core.node.port.PortObject;
import org.knime.core.node.port.PortObjectSpec;
import org.knime.core.node.util.CheckUtils;
import org.knime.core.node.workflow.capture.WorkflowPortObject;
import org.knime.core.node.workflow.capture.WorkflowPortObjectSpec;
import org.knime.core.node.workflow.capture.WorkflowSegment;
import org.knime.core.util.FileUtil;
import org.knime.filehandling.core.connections.FSConnection;
import org.knime.filehandling.core.connections.FSFileSystem;
import org.knime.filehandling.core.connections.FSFiles;
import org.knime.filehandling.core.connections.FSPath;
import org.knime.filehandling.core.connections.workflowaware.WorkflowAware;
import org.knime.filehandling.core.defaultnodesettings.filechooser.writer.WritePathAccessor;
import org.knime.filehandling.core.defaultnodesettings.status.NodeModelStatusConsumer;
import org.knime.filehandling.core.defaultnodesettings.status.StatusMessage.MessageType;
import org.knime.filehandling.core.port.FileSystemPortObject;
import org.knime.filehandling.core.port.FileSystemPortObjectSpec;

import com.knime.enterprise.client.filehandling.rest.RestFileSystem;
import com.knime.enterprise.server.rest.api.v4.repository.ent.Snapshot;

import jakarta.ws.rs.WebApplicationException;

/**
 * The "Deploy Workflow to Server" node model.
 *
 * @author Mark Ortmann, KNIME GmbH, Berlin, Germany
 */
final class DeployWorkflow3NodeModel extends NodeModel {

    /** The default workflow name. */
    private static final String DEFAULT_WORKFLOW_NAME = "workflow";

    private final DeployWorkflow3Config m_cfg;

    private final NodeModelStatusConsumer m_statusConsumer;

    private final int m_fsConnectionInputPortIndex;

    private final int m_workflowInputPortIndex;

    DeployWorkflow3NodeModel(final PortsConfiguration portsConfig) {
        super(portsConfig.getInputPorts(), portsConfig.getOutputPorts());
        m_cfg = new DeployWorkflow3Config(portsConfig);
        m_statusConsumer = new NodeModelStatusConsumer(EnumSet.of(MessageType.ERROR, MessageType.WARNING));

        // the port always exists and is unique therefore this cannot cause a NPE
        m_fsConnectionInputPortIndex =
            portsConfig.getInputPortLocation().get(DeployWorkflow3NodeFactory.FS_CONNECT_GRP_ID)[0];
        m_workflowInputPortIndex =
            portsConfig.getInputPortLocation().get(DeployWorkflow3NodeFactory.WORKFLOW_GRP_ID)[0];
    }

    @Override
    protected PortObjectSpec[] configure(final PortObjectSpec[] inSpecs) throws InvalidSettingsException {
        validateFSConnection((FileSystemPortObjectSpec)inSpecs[m_fsConnectionInputPortIndex]);
        m_cfg.getFileChooserModel().configureInModel(inSpecs, m_statusConsumer);
        m_statusConsumer.setWarningsIfRequired(this::setWarningMessage);
        m_cfg.getIOModel().validateSettings();
        return new PortObjectSpec[]{};
    }

    private static void validateFSConnection(final FileSystemPortObjectSpec spec) throws InvalidSettingsException {
        String specifier = spec.getFSLocationSpec().getFileSystemSpecifier() //
            .map(s -> StringUtils.substringBefore(s, ":")).orElse("<null>");
        String expectedLocation = "knime-server";
        CheckUtils.checkSetting(expectedLocation.equals(specifier),
            "Connected port does not represent a server "
                + "connection (detected type \"%s\" but expected \"%s\") -- is the input connected a "
                + "\"KNIME Server Connection\" node?",
            specifier, expectedLocation);

        @SuppressWarnings("resource")
        final FSConnection fsConnection = spec.getFileSystemConnection()
            .orElseThrow(() -> new InvalidSettingsException("No server connection available"));

        @SuppressWarnings("resource")
        FSFileSystem<?> fsSystem = fsConnection.getFileSystem();
        CheckUtils.checkSetting(fsSystem instanceof RestFileSystem,
            "Connected port does not represent a server connection (detected type \"%s\" but expected \"%s\") "
                + "-- is the input connected a \"KNIME Server Connection\" node?",
            specifier, expectedLocation);
    }

    @SuppressWarnings("resource")
    @Override
    protected PortObject[] execute(final PortObject[] inObjects, final ExecutionContext exec) throws Exception {
        try (final WritePathAccessor accessor = m_cfg.getFileChooserModel().createWritePathAccessor()) {
            final FSPath dir = accessor.getOutputPath(m_statusConsumer);
            m_statusConsumer.setWarningsIfRequired(this::setWarningMessage);

            exec.setProgress(.1, () -> "Checking if workflow group exists.");
            //check user settings and create missing folders if required
            createDirectoriesIfMissing(dir);

            WorkflowPortObject workflowPortObject = (WorkflowPortObject)inObjects[m_workflowInputPortIndex];
            final String workflowName = getWorkflowName(workflowPortObject.getSpec());
            final FSPath dest = (FSPath)dir.resolve(workflowName);
            exec.setProgress(.3, () -> "Checking if workflow exists.");
            validatePath(dest);

            WorkflowSegment segment = workflowPortObject.getSpec().getWorkflowSegment();
            if (m_cfg.getDoUpdateTemplateLinksModel().getBooleanValue()) {
                WorkflowSegmentManipulations.UPDATE_LINKED_TEMPLATES.apply(segment);
            }
            if (m_cfg.getDoRemoveTemplateLinksModel().getBooleanValue()) {
                WorkflowSegmentManipulations.REMOVE_TEMPLATE_LINKS.apply(segment);
            }

            exec.setProgress(.5, () -> "Saving workflow to disk.");
            final File tmpDir = FileUtil.createTempDir("deploy-workflow");
            final File localSource =
                WorkflowWriterNodeModel.write(tmpDir, workflowName, workflowPortObject.getSpec().getWorkflowSegment(),
                    exec, m_cfg.getIOModel(), true, workflowPortObject, false, this::setWarningMessage);

            exec.setProgress(.7, () -> "Deploying workflow onto KNIME Server.");
            final boolean overwrite;
            if (m_cfg.getExistsOption() == ExistsOption.OVERWRITE) {
                overwrite = true;
            } else {
                overwrite = false;
            }
            ((WorkflowAware)dest.getFileSystem().provider()).deployWorkflow(localSource.toPath(), dest, overwrite, false);
            FileUtil.deleteRecursively(tmpDir);

            if (m_cfg.createSnapshotModel().getBooleanValue()) {
                exec.setProgress(.9, () -> "Creating Snapshot.");
                createSnapshot(inObjects, dest.toAbsolutePath().toString());
            }
            return new PortObject[]{};
        }
    }

    private void createSnapshot(final PortObject[] inObjects, final String workflowPath)
        throws InvalidSettingsException, IOException {
        @SuppressWarnings("resource")
        final RestFileSystem restFS =
            (RestFileSystem)((FileSystemPortObject)inObjects[m_fsConnectionInputPortIndex]).getFileSystemConnection()
                .orElseThrow(() -> new InvalidSettingsException("No server connection available")).getFileSystem();
        try {
            final Snapshot snapshot = restFS.getRestClient().getRestServerContent().createSnapshot(workflowPath,
                m_cfg.getSnapshotNameModel().getStringValue());
            if (snapshot.getError() != null) {
                throw new IOException(
                    String.format("Snapshot could not be created: %s", snapshot.getError().getMessage()));
            }
        } catch (final WebApplicationException e) {
            throw new IllegalStateException(String.format("%s while creating snapshot: %s: ",
                e.getClass().getSimpleName(), e.getResponse().getStatus()), e);
        }
    }

    /**
     * Create the missing folders in provided output path, depends on the "Create missing folders" option in settings
     *
     * @param outputPath The FSPath for output folder
     * @throws IOException Throw exception if folder is missing and user has not checked "Create missing folders" option
     *             in settings
     */
    private void createDirectoriesIfMissing(final FSPath outputPath) throws IOException {
        if (!Files.exists(outputPath)) {
            if (m_cfg.getFileChooserModel().isCreateMissingFolders()) {
                FSFiles.createDirectories(outputPath);
            } else {
                throw new IOException(String.format(
                    "The directory '%s' does not exist and must not be created due to user settings.", outputPath));
            }
        }
    }

    private String getWorkflowName(final WorkflowPortObjectSpec spec) {
        if (m_cfg.useCustomWorkflowName()) {
            return m_cfg.getCustomWorkflowNameModel().getStringValue();
        } else {
            return getDefaultWorkflowName(spec);
        }
    }

    private void validatePath(final FSPath dest) throws InvalidSettingsException, IOException {
        if (m_cfg.getExistsOption() == ExistsOption.FAIL && FSFiles.exists(dest)) {
            throw new InvalidSettingsException(String
                .format("Destination path \"%s\" exists and must not be overwritten due to user settings.", dest));
        }
    }

    @Override
    protected void saveSettingsTo(final NodeSettingsWO settings) {
        m_cfg.saveSettingsInModel(settings);
    }

    @Override
    protected void validateSettings(final NodeSettingsRO settings) throws InvalidSettingsException {
        m_cfg.validateSettingsInModel(settings);
    }

    @Override
    protected void loadValidatedSettingsFrom(final NodeSettingsRO settings) throws InvalidSettingsException {
        m_cfg.loadSettingsInModel(settings);
    }

    public static final String getDefaultWorkflowName(final WorkflowPortObjectSpec spec) {
        final String name =
            FileUtil.ILLEGAL_FILENAME_CHARS_PATTERN.matcher(spec.getWorkflowName()).replaceAll("_").trim();
        if (!name.isEmpty()) {
            return name;
        }
        return DEFAULT_WORKFLOW_NAME;
    }

    @Override
    protected void loadInternals(final File nodeInternDir, final ExecutionMonitor exec) {
        // nothing to do
    }

    @Override
    protected void saveInternals(final File nodeInternDir, final ExecutionMonitor exec) {
        // nothing to do
    }

    @Override
    protected void reset() {
        // nothing to do
    }

}
