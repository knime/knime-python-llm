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

import java.util.Arrays;

import org.knime.ai.core.node.tool.workflow2tool.WorkflowToToolNodeSettings.OutputColumnPolicy;
import org.knime.core.data.DataTableSpec;
import org.knime.core.data.container.ColumnRearranger;
import org.knime.core.node.BufferedDataTable;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.context.ports.PortsConfiguration;
import org.knime.core.node.message.MessageBuilder;
import org.knime.core.node.port.PortObject;
import org.knime.core.node.port.PortObjectSpec;
import org.knime.core.webui.node.impl.WebUINodeModel;
import org.knime.filehandling.core.connections.FSConnection;
import org.knime.filehandling.core.port.FileSystemPortObjectSpec;

/**
 * @author Martin Horn, KNIME GmbH, Konstanz, Germany
 */
public final class WorkflowToToolNodeModel extends WebUINodeModel<WorkflowToToolNodeSettings> {

    private final int m_fileSystemConnectionPortIndex;

    private final int m_dataTablePortIndex;

    WorkflowToToolNodeModel(final PortsConfiguration portsConfig) {
        super(portsConfig.getInputPorts(), portsConfig.getOutputPorts(), WorkflowToToolNodeSettings.class);
        m_fileSystemConnectionPortIndex =
            getFirstPortIndexInGroup(portsConfig, WorkflowToToolNodeFactory.CONNECTION_INPUT_PORT_GRP_NAME);
        m_dataTablePortIndex =
            getFirstPortIndexInGroup(portsConfig, WorkflowToToolNodeFactory.PATH_TABLE_INPUT_PORT_GRP_NAME);
    }

    private static int getFirstPortIndexInGroup(final PortsConfiguration portsConfig, final String portGroupName) {
        final int[] portsInGroup = portsConfig.getInputPortLocation().get(portGroupName);
        if (portsInGroup != null && portGroupName.length() > 0) {
            return portsInGroup[0];
        } else {
            return -1;
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected PortObjectSpec[] configure(final PortObjectSpec[] inSpecs, final WorkflowToToolNodeSettings modelSettings)
        throws InvalidSettingsException {
        var messageBuilder = createMessageBuilder();
        var pathColumnIndex = getPathColumnIndex(inSpecs, modelSettings);
        try (var cellFactory =
            createCellFactory(inSpecs, null, messageBuilder, pathColumnIndex, getOutputColumnName(modelSettings))) {
            return new DataTableSpec[]{createColumnRearranger((DataTableSpec)inSpecs[m_dataTablePortIndex], cellFactory,
                modelSettings.m_outputColumnPolicy, pathColumnIndex).createSpec()};
        } finally {
            messageBuilder.build().ifPresent(m -> setWarning(m));
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected PortObject[] execute(final PortObject[] inData, final ExecutionContext exec,
        final WorkflowToToolNodeSettings modelSettings) throws Exception {
        var specs = Arrays.stream(inData).map(PortObject::getSpec).toArray(PortObjectSpec[]::new);
        var messageBuilder = createMessageBuilder();
        var pathColumnIndex = getPathColumnIndex(specs, modelSettings);
        try (var cellFactory =
            createCellFactory(specs, exec, messageBuilder, pathColumnIndex, getOutputColumnName(modelSettings))) {
            return new PortObject[]{exec.createColumnRearrangeTable((BufferedDataTable)inData[m_dataTablePortIndex],
                createColumnRearranger((DataTableSpec)specs[m_dataTablePortIndex], cellFactory,
                    modelSettings.m_outputColumnPolicy, pathColumnIndex),
                exec)};
        } finally {
            messageBuilder.build().ifPresent(m -> setWarning(m));
        }
    }

    private static String getOutputColumnName(final WorkflowToToolNodeSettings settings) {
        return settings.m_outputColumnPolicy == OutputColumnPolicy.REPLACE ? settings.m_inputColumnName
            : settings.m_outputColumnName;
    }

    private int getPathColumnIndex(final PortObjectSpec[] specs, final WorkflowToToolNodeSettings settings)
        throws InvalidSettingsException {
        if (settings.m_inputColumnName == null) {
            throw new InvalidSettingsException("No path column selected");
        }
        var spec = (DataTableSpec)specs[m_dataTablePortIndex];
        var pathColumnIndex = spec.findColumnIndex(settings.m_inputColumnName);
        if (pathColumnIndex == -1) {
            throw new InvalidSettingsException("No column found for name " + settings.m_inputColumnName);
        }
        return pathColumnIndex;
    }

    private record PathColumnIndexAndName(int index, String name) {
    }

    private PathToWorkflowCellFactory createCellFactory(final PortObjectSpec[] specs, final ExecutionContext exec,
        final MessageBuilder messageBuilder, final int pathColumnIndex, final String outputColumnName) {
        final FSConnection fsConnection =
            FileSystemPortObjectSpec.getFileSystemConnection(specs, m_fileSystemConnectionPortIndex).orElse(null);
        return new PathToWorkflowCellFactory(pathColumnIndex, outputColumnName, fsConnection, exec, messageBuilder);
    }

    private static ColumnRearranger createColumnRearranger(final DataTableSpec spec,
        final PathToWorkflowCellFactory cellFactory, final OutputColumnPolicy pathColumnPolicy,
        final int pathColumnIndex) {
        var rearranger = new ColumnRearranger(spec);
        if (pathColumnPolicy == OutputColumnPolicy.APPEND) {
            rearranger.append(cellFactory);
        } else {
            rearranger.replace(cellFactory, pathColumnIndex);
        }
        return rearranger;
    }

}
