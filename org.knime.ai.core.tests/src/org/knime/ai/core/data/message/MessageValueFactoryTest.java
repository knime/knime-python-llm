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
 *   May 23, 2025 (Adrian Nembach, KNIME GmbH, Konstanz, Germany): created
 */
package org.knime.ai.core.data.message;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import org.junit.jupiter.api.Test;
import org.knime.core.table.schema.ListDataSpec;
import org.knime.core.table.schema.StringDataSpec;
import org.knime.core.table.schema.StructDataSpec;
import org.knime.core.table.schema.VarBinaryDataSpec;

class MessageValueFactoryTest {

    @Test
    void testGetSpec() {
        MessageValueFactory factory = new MessageValueFactory();
        var spec = factory.getSpec();
        assertNotNull(spec);
        // Should be a StructDataSpec with 5 fields
        assertEquals(StructDataSpec.class, spec.getClass());
        var structSpec = (StructDataSpec) spec;
        assertEquals(5, structSpec.size());
        // Check types of fields
        assertEquals(StringDataSpec.class, structSpec.getDataSpec(0).getClass()); // type
        assertEquals(ListDataSpec.class, structSpec.getDataSpec(1).getClass());   // content
        assertEquals(ListDataSpec.class, structSpec.getDataSpec(2).getClass());   // tool calls
        assertEquals(StringDataSpec.class, structSpec.getDataSpec(3).getClass()); // tool call id
        assertEquals(StringDataSpec.class, structSpec.getDataSpec(4).getClass()); // name

        // Check content field (list of struct with 2 fields: String, VarBinary)
        var contentList = (ListDataSpec) structSpec.getDataSpec(1);
        var contentStruct = contentList.getInner();
        assertEquals(StructDataSpec.class, contentStruct.getClass());
        var contentStructSpec = (StructDataSpec) contentStruct;
        assertEquals(2, contentStructSpec.size());
        assertEquals(StringDataSpec.class, contentStructSpec.getDataSpec(0).getClass()); // content type
        assertEquals(VarBinaryDataSpec.class, contentStructSpec.getDataSpec(1).getClass()); // content data

        // Check tool calls field (list of struct with 3 fields: String, String, String)
        var toolCallsList = (ListDataSpec) structSpec.getDataSpec(2);
        var toolCallStruct = toolCallsList.getInner();
        assertEquals(StructDataSpec.class, toolCallStruct.getClass());
        var toolCallStructSpec = (StructDataSpec) toolCallStruct;
        assertEquals(3, toolCallStructSpec.size());
        assertEquals(StringDataSpec.class, toolCallStructSpec.getDataSpec(0).getClass()); // id
        assertEquals(StringDataSpec.class, toolCallStructSpec.getDataSpec(1).getClass()); // name
        assertEquals(StringDataSpec.class, toolCallStructSpec.getDataSpec(2).getClass()); // arguments
    }
}
