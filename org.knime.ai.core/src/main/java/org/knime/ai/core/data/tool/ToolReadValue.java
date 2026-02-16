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
 *   Feb 16, 2026 (Carsten Haubold, KNIME GmbH, Konstanz, Germany): created
 */
package org.knime.ai.core.data.tool;

import org.knime.core.data.v2.ReadValue;
import org.knime.core.table.access.ByteAccess.ByteReadAccess;
import org.knime.core.table.access.IntAccess.IntReadAccess;
import org.knime.core.table.access.StringAccess.StringReadAccess;
import org.knime.core.table.access.StructAccess.StructReadAccess;
import org.knime.core.table.access.VarBinaryAccess.VarBinaryReadAccess;

/**
 * ReadValue implementation for Tool (WorkflowTool and MCPTool).
 *
 * @author Carsten Haubold, KNIME GmbH, Konstanz, Germany
 */
final class ToolReadValue implements ReadValue {

    private final ByteReadAccess m_toolTypeAccess;
    private final StringReadAccess m_nameAccess;
    private final StringReadAccess m_descriptionAccess;
    private final StringReadAccess m_paramSchemaAccess;
    private final StringReadAccess m_inputSpecAccess;
    private final StringReadAccess m_outputSpecAccess;
    private final IntReadAccess m_messageOutputPortAccess;
    private final VarBinaryReadAccess m_filestoreAccess;
    private final StringReadAccess m_serverUriAccess;

    ToolReadValue(final StructReadAccess access) {
        m_toolTypeAccess = access.getAccess(0);
        m_nameAccess = access.getAccess(1);
        m_descriptionAccess = access.getAccess(2);
        m_paramSchemaAccess = access.getAccess(3);
        m_inputSpecAccess = access.getAccess(4);
        m_outputSpecAccess = access.getAccess(5);
        m_messageOutputPortAccess = access.getAccess(6);
        m_filestoreAccess = access.getAccess(7);
        m_serverUriAccess = access.getAccess(8);
    }

    /**
     * @return the tool type (WORKFLOW=0, MCP=1)
     */
    public ToolType getToolType() {
        return ToolType.fromIndex(m_toolTypeAccess.getByteValue());
    }

    /**
     * @return the tool name
     */
    public String getName() {
        return m_nameAccess.getStringValue();
    }

    /**
     * @return the tool description
     */
    public String getDescription() {
        return m_descriptionAccess.getStringValue();
    }

    /**
     * @return the parameter schema as JSON string
     */
    public String getParameterSchema() {
        return m_paramSchemaAccess.isMissing() ? null : m_paramSchemaAccess.getStringValue();
    }

    /**
     * @return the input spec as JSON string (Workflow only)
     */
    public String getInputSpec() {
        return m_inputSpecAccess.isMissing() ? null : m_inputSpecAccess.getStringValue();
    }

    /**
     * @return the output spec as JSON string (Workflow only)
     */
    public String getOutputSpec() {
        return m_outputSpecAccess.isMissing() ? null : m_outputSpecAccess.getStringValue();
    }

    /**
     * @return the message output port index (Workflow only, -1 for MCP)
     */
    public int getMessageOutputPortIndex() {
        return m_messageOutputPortAccess.getIntValue();
    }

    /**
     * @return the filestore keys (Workflow only)
     */
    public Object getFilestoreKeys() {
        return m_filestoreAccess.isMissing() ? null : m_filestoreAccess.getObject();
    }

    /**
     * @return the MCP server URI (MCP only)
     */
    public String getServerUri() {
        return m_serverUriAccess.isMissing() ? null : m_serverUriAccess.getStringValue();
    }
}
