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

import org.knime.core.data.v2.WriteValue;
import org.knime.core.table.access.ByteAccess.ByteWriteAccess;
import org.knime.core.table.access.IntAccess.IntWriteAccess;
import org.knime.core.table.access.StringAccess.StringWriteAccess;
import org.knime.core.table.access.StructAccess.StructWriteAccess;
import org.knime.core.table.access.VarBinaryAccess.VarBinaryWriteAccess;

/**
 * WriteValue implementation for Tool (WorkflowTool and MCPTool).
 *
 * @author Carsten Haubold, KNIME GmbH, Konstanz, Germany
 */
final class ToolWriteValue implements WriteValue<ToolReadValue> {

    private final ByteWriteAccess m_toolTypeAccess;
    private final StringWriteAccess m_nameAccess;
    private final StringWriteAccess m_descriptionAccess;
    private final StringWriteAccess m_paramSchemaAccess;
    private final StringWriteAccess m_inputSpecAccess;
    private final StringWriteAccess m_outputSpecAccess;
    private final IntWriteAccess m_messageOutputPortAccess;
    private final VarBinaryWriteAccess m_filestoreAccess;
    private final StringWriteAccess m_serverUriAccess;

    ToolWriteValue(final StructWriteAccess access) {
        m_toolTypeAccess = access.getWriteAccess(0);
        m_nameAccess = access.getWriteAccess(1);
        m_descriptionAccess = access.getWriteAccess(2);
        m_paramSchemaAccess = access.getWriteAccess(3);
        m_inputSpecAccess = access.getWriteAccess(4);
        m_outputSpecAccess = access.getWriteAccess(5);
        m_messageOutputPortAccess = access.getWriteAccess(6);
        m_filestoreAccess = access.getWriteAccess(7);
        m_serverUriAccess = access.getWriteAccess(8);
    }

    /**
     * Set the tool type.
     * 
     * @param toolType the tool type (WORKFLOW or MCP)
     */
    public void setToolType(final ToolType toolType) {
        m_toolTypeAccess.setByteValue(toolType.getIndex());
    }

    /**
     * Set the tool name.
     * 
     * @param name the tool name
     */
    public void setName(final String name) {
        m_nameAccess.setStringValue(name);
    }

    /**
     * Set the tool description.
     * 
     * @param description the tool description
     */
    public void setDescription(final String description) {
        m_descriptionAccess.setStringValue(description);
    }

    /**
     * Set the parameter schema as JSON string.
     * 
     * @param parameterSchema the parameter schema JSON
     */
    public void setParameterSchema(final String parameterSchema) {
        if (parameterSchema == null) {
            m_paramSchemaAccess.setMissing();
        } else {
            m_paramSchemaAccess.setStringValue(parameterSchema);
        }
    }

    /**
     * Set the input spec as JSON string (Workflow only).
     * 
     * @param inputSpec the input spec JSON
     */
    public void setInputSpec(final String inputSpec) {
        if (inputSpec == null) {
            m_inputSpecAccess.setMissing();
        } else {
            m_inputSpecAccess.setStringValue(inputSpec);
        }
    }

    /**
     * Set the output spec as JSON string (Workflow only).
     * 
     * @param outputSpec the output spec JSON
     */
    public void setOutputSpec(final String outputSpec) {
        if (outputSpec == null) {
            m_outputSpecAccess.setMissing();
        } else {
            m_outputSpecAccess.setStringValue(outputSpec);
        }
    }

    /**
     * Set the message output port index (Workflow only).
     * 
     * @param index the port index (-1 for MCP tools)
     */
    public void setMessageOutputPortIndex(final int index) {
        m_messageOutputPortAccess.setIntValue(index);
    }

    /**
     * Set the filestore keys (Workflow only).
     * 
     * @param filestoreKeys the filestore keys object
     */
    public void setFilestoreKeys(final Object filestoreKeys) {
        if (filestoreKeys == null) {
            m_filestoreAccess.setMissing();
        } else {
            m_filestoreAccess.setObject(filestoreKeys);
        }
    }

    /**
     * Set the MCP server URI (MCP only).
     * 
     * @param serverUri the server URI
     */
    public void setServerUri(final String serverUri) {
        if (serverUri == null) {
            m_serverUriAccess.setMissing();
        } else {
            m_serverUriAccess.setStringValue(serverUri);
        }
    }

    @Override
    public void setValue(final ToolReadValue value) {
        setToolType(value.getToolType());
        setName(value.getName());
        setDescription(value.getDescription());
        setParameterSchema(value.getParameterSchema());
        setInputSpec(value.getInputSpec());
        setOutputSpec(value.getOutputSpec());
        setMessageOutputPortIndex(value.getMessageOutputPortIndex());
        setFilestoreKeys(value.getFilestoreKeys());
        setServerUri(value.getServerUri());
    }
}
