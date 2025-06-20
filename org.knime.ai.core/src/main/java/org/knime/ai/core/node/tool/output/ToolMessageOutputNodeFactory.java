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
package org.knime.ai.core.node.tool.output;

import org.knime.core.webui.node.impl.WebUINodeConfiguration;
import org.knime.core.webui.node.impl.WebUINodeFactory;

/**
 * @author Martin Horn, KNIME GmbH, Konstanz, Germany
 */
@SuppressWarnings("restriction")
public final class ToolMessageOutputNodeFactory extends WebUINodeFactory<ToolMessageOutputNodeModel> {

    private static final String DESCRIPTION = """
            <p>
            This node allows specifying the output message of the Tool, which is shown
            to the AI agent that invoked it.
            </p>

            <p>
            Only the content of the <b>first cell</b> of the input table (first column, first row)
            is forwarded to the agent. If the Tool workflow does not specify an output
            message, a placeholder message notifying the agent about successful Tool execution
            will be used instead.
            </p>

            <p>
            This node belongs to the <i>communication layer</i> of the Tool, and is not meant to transmit
            data. In order to pass data into or out of the Tool, add <i>Workflow Input</i>
            and/or <i>Workflow Output</i> nodes. These facilitate the <i>data layer</i> and can be combined with this
            node as needed.
            </p>
            """;

    private static final WebUINodeConfiguration CONFIG = WebUINodeConfiguration.builder() //
        .name("Tool Message Output") //
        .icon("./Tool-message-output.png") //
        .shortDescription("This node allows specifying the output message of the Tool.") //
        .fullDescription(DESCRIPTION) //
        .modelSettingsClass(ToolMessageOutputNodeSettings.class) //
        .addInputTable("Tool message", "Table containing the tool message in the first cell.") //
        .nodeType(NodeType.Container) //
        .sinceVersion(5, 5, 0) //
        .build();

    public ToolMessageOutputNodeFactory() {
        super(CONFIG);
    }

    @Override
    public ToolMessageOutputNodeModel createNodeModel() {
        return new ToolMessageOutputNodeModel(CONFIG);
    }

}
