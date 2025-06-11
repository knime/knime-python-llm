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
 *   May 21, 2025 (hornm): created
 */
package org.knime.ai.core.node.tool.workflow2tool;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.zip.ZipInputStream;

import org.knime.ai.core.node.tool.output.ToolMessageOutputNodeModel;
import org.knime.core.data.DataCell;
import org.knime.core.data.DataColumnSpec;
import org.knime.core.data.DataColumnSpecCreator;
import org.knime.core.data.DataRow;
import org.knime.core.data.DataType;
import org.knime.core.data.MissingCell;
import org.knime.core.data.container.SingleCellFactory;
import org.knime.core.node.CanceledExecutionException;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.KNIMEException;
import org.knime.core.node.NodeLogger;
import org.knime.core.node.agentic.tool.ToolWorkflowMetadata;
import org.knime.core.node.agentic.tool.WorkflowToolCell;
import org.knime.core.node.agentic.tool.WorkflowToolCell.ToolIncompatibleWorkflowException;
import org.knime.core.node.message.Message;
import org.knime.core.node.message.MessageBuilder;
import org.knime.core.node.workflow.UnsupportedWorkflowVersionException;
import org.knime.core.node.workflow.WorkflowManager;
import org.knime.core.node.workflow.WorkflowPersistor.LoadResultEntry.LoadResultEntryType;
import org.knime.core.node.workflow.WorkflowPersistor.WorkflowLoadResult;
import org.knime.core.node.workflow.capture.BuildWorkflowsUtil;
import org.knime.core.node.workflow.capture.WorkflowSegment;
import org.knime.core.util.FileUtil;
import org.knime.core.util.LockFailedException;
import org.knime.core.util.hub.ItemVersion;
import org.knime.filehandling.core.connections.FSConnection;
import org.knime.filehandling.core.connections.FSFiles;
import org.knime.filehandling.core.connections.FSPath;
import org.knime.filehandling.core.connections.location.MultiFSPathProviderFactory;
import org.knime.filehandling.core.connections.workflowaware.Entity;
import org.knime.filehandling.core.connections.workflowaware.WorkflowAwareUtil;
import org.knime.filehandling.core.data.location.FSLocationValue;
import org.knime.filehandling.core.util.TempPathCloseable;

/**
 * @author Martin Horn, KNIME GmbH, Konstanz, Germany
 */
final class PathToWorkflowCellFactory extends SingleCellFactory implements Closeable {

    private final int m_pathColumnIndex;

    private final MultiFSPathProviderFactory m_multiFSPathProviderFactory;

    private final ExecutionContext m_exec;

    private final MessageBuilder m_messageBuilder;

    PathToWorkflowCellFactory(final int pathColumnIndex, final String outputColumnName, final FSConnection fsConnection,
        final ExecutionContext exec, final MessageBuilder messageBuilder) {
        super(createColumnSpec(outputColumnName));
        m_pathColumnIndex = pathColumnIndex;
        m_exec = exec;
        m_messageBuilder = messageBuilder;
        m_multiFSPathProviderFactory = new MultiFSPathProviderFactory(fsConnection);
    }

    @Override
    public DataCell getCell(final DataRow row, final long rowIndex) {
        var cell = row.getCell(m_pathColumnIndex);
        if (cell.isMissing()) {
            m_messageBuilder.addRowIssue(m_pathColumnIndex, rowIndex, "Missing value");
            return DataType.getMissingCell();
        }
        var fsLocation = ((FSLocationValue)cell).getFSLocation();
        try (@SuppressWarnings("resource")
        var pathProvider =
            m_multiFSPathProviderFactory.getOrCreateFSPathProviderFactory(fsLocation).create(fsLocation)) {
            final var fsPath = pathProvider.getPath();
            if (!FSFiles.exists(fsPath)) {
                var message = "Path does not exist: " + fsPath;
                m_messageBuilder.addRowIssue(m_pathColumnIndex, rowIndex, message + ". Missing value added.");
                return new MissingCell(message);
            }
            var wfTempPath = toLocalWorkflowDir(fsPath, null);
            var wfFile = wfTempPath.getTempFileOrFolder();
            var wfm = readWorkflow(wfFile.toFile(), m_exec.createSilentSubExecutionContext(0),
                m_messageBuilder);
            wfm.setName(fsPath.getFileName().toString());
            if (resetWorkflow(wfm)) {
                m_messageBuilder.addRowIssue(m_pathColumnIndex, rowIndex,
                    "Workflow was (partially) executed and has been reset.");
            }
            try {
                var toolMessageOutputNode = wfm.findNodes(ToolMessageOutputNodeModel.class, false);
                if (toolMessageOutputNode.size() > 1) {
                    var message = "Multiple tool message output nodes found in workflow";
                    m_messageBuilder.addRowIssue(m_pathColumnIndex, rowIndex, message);
                    return new MissingCell(message);
                }
                var dataAreaPath = wfFile.resolve("data");
                if (!FSFiles.exists(dataAreaPath)) {
                    dataAreaPath = null;
                }
                return WorkflowToolCell.createFromAndModifyWorkflow(wfm,
                    new ToolWorkflowMetadata(
                        toolMessageOutputNode.isEmpty() ? null : toolMessageOutputNode.keySet().iterator().next()),
                    dataAreaPath);
            } catch (ToolIncompatibleWorkflowException e) {
                var message = "Workflow can't be turned into a tool: " + e.getMessage();
                m_messageBuilder.addRowIssue(m_pathColumnIndex, rowIndex, message);
                return new MissingCell(message);
            } finally {
                wfTempPath.close();
            }
        } catch (IOException | InvalidSettingsException | CanceledExecutionException
                | UnsupportedWorkflowVersionException | LockFailedException | KNIMEException e) {
            var message = "Unable to read workflow: " + e.getMessage();
            NodeLogger.getLogger(WorkflowToToolNodeFactory.class).warn(message, e);
            m_messageBuilder.addRowIssue(m_pathColumnIndex, rowIndex, message + ". Missing value added.");
            return new MissingCell(message);
        }
    }

    @Override
    public void afterProcessing() {
        if (m_messageBuilder.getIssueCount() > 0) {
            m_messageBuilder.withSummary("Issues while reading workflows.");
        }
    }

    @SuppressWarnings("resource")
    private static TempPathCloseable toLocalWorkflowDir(final FSPath path, final ItemVersion version)
        throws IOException {
        // the connected file system is either WorkflowAware or provides the workflow as a '.knwf'-file

        ensureIsWorkflow(path);

        final var fs = path.getFileSystem();

        // Try with version
        final var wfvAware = fs.getItemVersionAware();
        if (wfvAware.isPresent() && version != null) {
            return wfvAware.get().downloadWorkflowAtVersion(path, version);
        }

        // Try to fetch workflow
        final var wfAware = fs.getWorkflowAware();
        if (wfAware.isPresent()) {
            return wfAware.get().toLocalWorkflowDir(path);
        }

        // Try plain input stream
        try (var in = FSFiles.newInputStream(path)) {
            return unzipToLocalDir(in);
        }
    }

    private static void ensureIsWorkflow(final FSPath path) throws IOException {
        final boolean isWorkflow;
        if (WorkflowAwareUtil.isWorkflowAwarePath(path)) {
            isWorkflow = WorkflowAwareUtil.getWorkflowAwareEntityOf(path) //
                .map(Entity.WORKFLOW::equals) //
                .orElse(false);
        } else {
            isWorkflow = path.toString().endsWith(".knwf");
        }

        if (!isWorkflow) {
            throw new IOException("Not a workflow");
        }
    }

    private static TempPathCloseable unzipToLocalDir(final InputStream in) throws IOException {
        File tmpDir = null;
        try (var zip = new ZipInputStream(in)) {
            tmpDir = FileUtil.createTempDir("workflow_to_tool");
            FileUtil.unzip(zip, tmpDir, 1);
        }
        return new TempPathCloseable(tmpDir.toPath());
    }

    private static WorkflowManager readWorkflow(final File wfFile, final ExecutionContext exec,
        final MessageBuilder messageBuilder) throws IOException, InvalidSettingsException, CanceledExecutionException,
        UnsupportedWorkflowVersionException, LockFailedException, KNIMEException {

        final var loadHelper = WorkflowSegment.createWorkflowLoadHelper(wfFile, messageBuilder::addTextIssue);
        final WorkflowLoadResult loadResult =
            WorkflowManager.EXTRACTED_WORKFLOW_ROOT.load(wfFile, exec, loadHelper, false);

        final var m = loadResult.getWorkflowManager();
        if (m == null) {
            throw KNIMEException.of( //
                Message.builder() //
                    .withSummary("Errors reading workflow.")
                    .addTextIssue(loadResult.getFilteredError("", LoadResultEntryType.Warning)) //
                    .build().orElseThrow());
        } else {
            try {
                var loadWarningOptional = BuildWorkflowsUtil.checkLoadResult(loadResult);
                loadWarningOptional.ifPresent(messageBuilder::addTextIssue);
            } catch (KNIMEException e) {
                WorkflowManager.EXTRACTED_WORKFLOW_ROOT.removeNode(m.getID());
                throw e;
            }
        }
        return loadResult.getWorkflowManager();

    }

    private static boolean resetWorkflow(final WorkflowManager wfm) {
        final var partiallyExecuted = wfm.canResetAll() && !wfm.getNodeContainers().isEmpty();
        if (partiallyExecuted) {
            wfm.resetAndConfigureAll();
        }
        return partiallyExecuted;
    }

    private static DataColumnSpec createColumnSpec(final String outputColumnName) {
        return new DataColumnSpecCreator(outputColumnName, WorkflowToolCell.TYPE).createSpec();
    }

    @Override
    public void close() {
        m_multiFSPathProviderFactory.close();
    }

}
