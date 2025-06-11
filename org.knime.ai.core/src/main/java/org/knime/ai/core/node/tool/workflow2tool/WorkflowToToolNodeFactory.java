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
 *   May 5, 2025 (hornm): created
 */
package org.knime.ai.core.node.tool.workflow2tool;

import java.util.Optional;

import org.knime.core.node.BufferedDataTable;
import org.knime.core.node.ConfigurableNodeFactory;
import org.knime.core.node.NodeDescription;
import org.knime.core.node.NodeDialogPane;
import org.knime.core.node.NodeView;
import org.knime.core.node.context.NodeCreationConfiguration;
import org.knime.core.webui.node.dialog.NodeDialog;
import org.knime.core.webui.node.dialog.NodeDialogFactory;
import org.knime.core.webui.node.dialog.SettingsType;
import org.knime.core.webui.node.dialog.defaultdialog.DefaultNodeDialog;
import org.knime.core.webui.node.impl.WebUINodeConfiguration;
import org.knime.core.webui.node.impl.WebUINodeFactory;
import org.knime.filehandling.core.port.FileSystemPortObject;

/**
 * @author Martin Horn, KNIME GmbH, Konstanz, Germany
 */
public final class WorkflowToToolNodeFactory extends ConfigurableNodeFactory<WorkflowToToolNodeModel>
    implements NodeDialogFactory {

    static final String CONNECTION_INPUT_PORT_GRP_NAME = "File System Connection";

    static final String PATH_TABLE_INPUT_PORT_GRP_NAME = "Paths";

    private static final String TOOL_TABLE_OUTPUT_PORT_GRP_NAME = "Tools";

    private static final String DESCRIPTION = """
            This node reads workflows from the provided paths and turns them into the tool data type.
            If the workflow can't be read, a missing value will be output instead. Workflows can't be read, e.g.,
            because the path doesn't reference a workflow or the workflow doesn't comply with tool conventions.
            Learn more about how to create a tool workflow here (TODO).
            Please note that (partially) executed workflows will stored in a reset state.
            """;

    private static final WebUINodeConfiguration CONFIG = WebUINodeConfiguration.builder() //
        .name("Workflow To Tool") //
        .icon("./Path-to-string.png") //
        .shortDescription("Turns workflow paths into tools.") //
        .fullDescription(DESCRIPTION) //
        .modelSettingsClass(WorkflowToToolNodeSettings.class) //
        .addInputPort(CONNECTION_INPUT_PORT_GRP_NAME, FileSystemPortObject.TYPE, "The file system connection", true)
        .addInputTable(PATH_TABLE_INPUT_PORT_GRP_NAME, "The input table containg the paths") //
        .addOutputPort(TOOL_TABLE_OUTPUT_PORT_GRP_NAME, BufferedDataTable.TYPE, "The output table with the tools") //
        .nodeType(NodeType.Manipulator) //
        .sinceVersion(5, 5, 0) //
        .build();

    @Override
    protected NodeDescription createNodeDescription() {
        return WebUINodeFactory.createNodeDescription(CONFIG);
    }

    @Override
    protected Optional<PortsConfigurationBuilder> createPortsConfigBuilder() {
        final PortsConfigurationBuilder builder = new PortsConfigurationBuilder();
        builder.addOptionalInputPortGroup(CONNECTION_INPUT_PORT_GRP_NAME, FileSystemPortObject.TYPE);
        builder.addFixedInputPortGroup(PATH_TABLE_INPUT_PORT_GRP_NAME, BufferedDataTable.TYPE);
        builder.addFixedOutputPortGroup(TOOL_TABLE_OUTPUT_PORT_GRP_NAME, BufferedDataTable.TYPE);
        return Optional.of(builder);
    }

    @Override
    protected WorkflowToToolNodeModel createNodeModel(final NodeCreationConfiguration creationConfig) {
        var portsConfig = creationConfig.getPortConfig().orElseThrow(IllegalStateException::new);
        return new WorkflowToToolNodeModel(portsConfig);
    }

    @Override
    protected boolean hasDialog() {
        return false;
    }

    @Override
    public NodeDialog createNodeDialog() {
        return new DefaultNodeDialog(SettingsType.MODEL, WorkflowToToolNodeSettings.class);
    }

    @Override
    protected NodeDialogPane createNodeDialogPane(final NodeCreationConfiguration creationConfig) {
        return null;
    }

    @Override
    protected int getNrNodeViews() {
        return 0;
    }

    @Override
    public NodeView<WorkflowToToolNodeModel> createNodeView(final int viewIndex,
        final WorkflowToToolNodeModel nodeModel) {
        return null;
    }

}
