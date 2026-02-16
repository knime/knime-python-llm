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
import org.knime.core.data.v2.ValueFactory;
import org.knime.core.data.v2.WriteValue;
import org.knime.core.data.v2.filestore.FileStoreValueFactory;
import org.knime.core.table.access.StructAccess.StructReadAccess;
import org.knime.core.table.access.StructAccess.StructWriteAccess;
import org.knime.core.table.schema.ByteDataSpec;
import org.knime.core.table.schema.DataSpec;
import org.knime.core.table.schema.IntDataSpec;
import org.knime.core.table.schema.StringDataSpec;
import org.knime.core.table.schema.StructDataSpec;
import org.knime.core.table.schema.VarBinaryDataSpec;

/**
 * Unified ValueFactory for reading and writing Tool instances (both WorkflowTool and MCPTool).
 * 
 * Uses a discriminator pattern with byte-indexed enum to support both tool types in a single column.
 * 
 * Arrow schema (9 fields):
 * - Field 0: tool_type (BYTE) - Enum index: 0=WORKFLOW, 1=MCP
 * - Field 1: name (STRING) - Common
 * - Field 2: description (STRING) - Common  
 * - Field 3: parameter_schema (STRING) - Common (JSON)
 * - Field 4: input_spec (STRING) - Workflow ports JSON or null
 * - Field 5: output_spec (STRING) - Workflow ports JSON or null
 * - Field 6: message_output_port_index (INT) - Workflow only (-1 for MCP)
 * - Field 7: workflow_filestore (FILESTORE) - Workflow only (null for MCP)
 * - Field 8: server_uri (STRING) - MCP only (null for Workflow)
 *
 * @author Carsten Haubold, KNIME GmbH, Konstanz, Germany
 */
public class ToolValueFactory implements ValueFactory<StructReadAccess, StructWriteAccess>, FileStoreValueFactory<StructReadAccess, StructWriteAccess> {

    @Override
    public DataSpec getSpec() {
        return new StructDataSpec(//
            ByteDataSpec.INSTANCE,      // 0: tool_type
            StringDataSpec.INSTANCE,    // 1: name
            StringDataSpec.INSTANCE,    // 2: description
            StringDataSpec.INSTANCE,    // 3: parameter_schema (JSON)
            StringDataSpec.INSTANCE,    // 4: input_spec (JSON, Workflow only)
            StringDataSpec.INSTANCE,    // 5: output_spec (JSON, Workflow only)
            IntDataSpec.INSTANCE,       // 6: message_output_port_index
            VarBinaryDataSpec.varbinary(), // 7: workflow_filestore
            StringDataSpec.INSTANCE     // 8: server_uri (MCP only)
        );
    }

    @Override
    public ReadValue createReadValue(final StructReadAccess access) {
        return new ToolReadValue(access);
    }

    @Override
    public WriteValue<?> createWriteValue(final StructWriteAccess access) {
        return new ToolWriteValue(access);
    }
}
