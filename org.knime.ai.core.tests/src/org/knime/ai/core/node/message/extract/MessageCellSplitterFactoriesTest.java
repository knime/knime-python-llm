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
package org.knime.ai.core.node.message.extract;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Optional;

import org.junit.jupiter.api.Test;
import org.knime.ai.core.data.message.MessageValue.ToolCall;
import org.knime.core.data.DataCell;
import org.knime.core.data.DataColumnSpec;
import org.knime.core.data.def.StringCell;
import org.knime.core.data.json.JSONCellFactory;

@SuppressWarnings("static-method")
final class MessageCellSplitterFactoriesTest {

    @Test
    void testCreateCellSplitterFactories_emptySettings() {
        MessagePartExtractorSettings settings = new MessagePartExtractorSettings();
        settings.m_imagePartsPrefix = Optional.empty();
        settings.m_textPartsPrefix = Optional.empty();
        settings.m_toolCallsPrefix = Optional.empty();
        settings.m_roleColumnName = Optional.empty();
        settings.m_toolCallIdColumnName = Optional.empty();
        settings.m_nameColumnName = Optional.empty();
        List<MessageCellSplitterFactories.CellSplitterFactory<?>> factories =
                MessageCellSplitterFactories.createCellSplitterFactories(settings, 0);
        assertNotNull(factories);
        assertTrue(factories.isEmpty());
    }

    @Test
    void testCreateColumnSpecCreator() {
        var creator = MessageCellSplitterFactories.createColumnSpecCreator("prefix", StringCell.TYPE);
        DataColumnSpec spec = creator.apply(0);
        assertEquals("prefix1", spec.getName());
        assertEquals(StringCell.TYPE, spec.getType());
    }

    @Test
    void testCreateContentCounter() {
        var msg = TestUtil.createMessageWithTextParts("foo", "bar");
        DataCell cell = msg;
        var counter = MessageCellSplitterFactories.createContentCounter("text");
        assertEquals(2, counter.apply(cell));
    }

    @Test
    void testCountToolCalls() {
        var msg = TestUtil.createMessageWithToolCalls(3);
        DataCell cell = msg;
        assertEquals(3, MessageCellSplitterFactories.countToolCalls(cell));
    }

    @Test
    void testExtractToolCalls() {
        var msg = TestUtil.createMessageWithToolCalls(2);
        DataCell[] cells = MessageCellSplitterFactories.extractToolCalls(msg);
        assertEquals(2, cells.length);
        for (DataCell cell : cells) {
            assertEquals(JSONCellFactory.TYPE, cell.getType());
        }
    }

    @Test
    void testPadWithMissing() {
        DataCell[] arr = { new StringCell("a") };
        DataCell[] padded = MessageCellSplitterFactories.padWithMissing(arr, 3);
        assertEquals(3, padded.length);
        assertEquals("a", ((StringCell)padded[0]).getStringValue());
        assertTrue(padded[1].isMissing());
        assertTrue(padded[2].isMissing());
    }

    @Test
    void testExtractRole() {
        var msg = TestUtil.createMessageWithRole("USER");
        StringCell cell = MessageCellSplitterFactories.extractRole(msg);
        assertEquals("USER", cell.getStringValue());
    }

    @Test
    void testExtractToolCallId() {
        var msg = TestUtil.createMessageWithToolCallId("id123");
        DataCell cell = MessageCellSplitterFactories.extractToolCallId(msg);
        assertEquals("id123", ((StringCell)cell).getStringValue());
    }

    @Test
    void testExtractName() {
        var msg = TestUtil.createMessageWithName("foo");
        DataCell cell = MessageCellSplitterFactories.extractName(msg);
        assertEquals("foo", ((StringCell)cell).getStringValue());
    }

    @Test
    void testToJsonCell() {
        var toolCall = new ToolCall("tool", "id", "{}\n");
        DataCell cell = MessageCellSplitterFactories.toJsonCell(toolCall);
        assertEquals(JSONCellFactory.TYPE, cell.getType());
    }
}
