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
package org.knime.python.llm.java.tool;

import java.io.Closeable;
import java.io.IOException;

import org.knime.python.llm.java.reader.WorkflowReaderNodeModel;
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
import org.knime.core.node.message.MessageBuilder;
import org.knime.core.node.workflow.UnsupportedWorkflowVersionException;
import org.knime.core.node.workflow.WorkflowManager;
import org.knime.core.util.LockFailedException;
import org.knime.filehandling.core.connections.FSConnection;
import org.knime.filehandling.core.connections.FSFiles;
import org.knime.filehandling.core.connections.location.MultiFSPathProviderFactory;
import org.knime.filehandling.core.data.location.FSLocationValue;

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
            var wfTempFolder = WorkflowReaderNodeModel.toLocalWorkflowDir(fsPath, null);
            var wfm = WorkflowReaderNodeModel.readWorkflow(wfTempFolder.getTempFileOrFolder().toFile(),
                m_exec.createSilentSubExecutionContext(0), m_messageBuilder);
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
                return WorkflowToolCell.createFromAndModifyWorkflow(wfm, new ToolWorkflowMetadata(
                    toolMessageOutputNode.isEmpty() ? null : toolMessageOutputNode.keySet().iterator().next()));
            } catch (ToolIncompatibleWorkflowException e) {
                var message = "Workflow can't be turned into a tool: " + e.getMessage();
                m_messageBuilder.addRowIssue(m_pathColumnIndex, rowIndex, message);
                return new MissingCell(message);
            } finally {
                wfTempFolder.close();
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
