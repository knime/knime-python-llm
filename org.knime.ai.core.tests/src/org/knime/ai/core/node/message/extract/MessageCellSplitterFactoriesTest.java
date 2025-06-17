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
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Optional;

import org.junit.jupiter.api.Test;
import org.knime.ai.core.data.message.MessageValue.MessageContentPart.MessageContentPartType;
import org.knime.ai.core.data.message.MessageValue.ToolCall;
import org.knime.core.data.DataCell;
import org.knime.core.data.DataColumnSpec;
import org.knime.core.data.DataColumnSpecCreator;
import org.knime.core.data.DataRow;
import org.knime.core.data.DataTableSpec;
import org.knime.core.data.RowKey;
import org.knime.core.data.def.DefaultRow;
import org.knime.core.data.def.StringCell;
import org.knime.core.data.json.JSONCellFactory;
import org.knime.core.node.InvalidSettingsException;

@SuppressWarnings("static-method")
final class MessageCellSplitterFactoriesTest {

    final static DataTableSpec SPEC =
        new DataTableSpec(new DataColumnSpecCreator("role", StringCell.TYPE).createSpec(),
            new DataColumnSpecCreator("name", StringCell.TYPE).createSpec(),
            new DataColumnSpecCreator("text1", StringCell.TYPE).createSpec());

    @Test
    void testCreateCellSplitterFactories_emptySettings() throws InvalidSettingsException {
        MessagePartExtractorSettings settings = new MessagePartExtractorSettings();
        settings.m_imagePartsPrefix = Optional.empty();
        settings.m_textPartsPrefix = Optional.empty();
        settings.m_toolCallsPrefix = Optional.empty();
        settings.m_roleColumnName = Optional.empty();
        settings.m_toolCallIdColumnName = Optional.empty();
        settings.m_nameColumnName = Optional.empty();
        List<MessageCellSplitterFactories.CellSplitterFactory<?>> factories =
                MessageCellSplitterFactories.createCellSplitterFactories(settings, 0, SPEC);
        assertNotNull(factories);
        assertTrue(factories.isEmpty());
    }

    @Test
    void testCreateCellSplitterFactories_throwsOnRoleColumnNameBlank() {
        MessagePartExtractorSettings settings = new MessagePartExtractorSettings();
        settings.m_roleColumnName = Optional.of("   ");

        InvalidSettingsException thrown = assertThrows(InvalidSettingsException.class, () -> {
            MessageCellSplitterFactories.createCellSplitterFactories(settings, 0, SPEC);
        });
        assertEquals("Role column name cannot be empty when 'Role column name' is enabled.", thrown.getMessage());
    }

    @Test
    void testCreateCellSplitterFactories_throwsOnNameColumnNameEmpty() {
        MessagePartExtractorSettings settings = new MessagePartExtractorSettings();
        settings.m_nameColumnName = Optional.of("");

        InvalidSettingsException thrown = assertThrows(InvalidSettingsException.class, () -> {
            MessageCellSplitterFactories.createCellSplitterFactories(settings, 0, SPEC);
        });
        assertEquals("Name column name cannot be empty when 'Name column name' is enabled.", thrown.getMessage());
    }

    @Test
    void testCreateCellSplitterFactories_throwsOnTextPartsPrefixBlank() {
        MessagePartExtractorSettings settings = new MessagePartExtractorSettings();
        settings.m_textPartsPrefix = Optional.of("\t"); // Set to blank

        InvalidSettingsException thrown = assertThrows(InvalidSettingsException.class, () -> {
            MessageCellSplitterFactories.createCellSplitterFactories(settings, 0, SPEC);
        });
        assertEquals("Text parts column prefix cannot be empty when 'Text parts column prefix' is enabled.", thrown.getMessage());
    }

    @Test
    void testCreateCellSplitterFactories_throwsOnImagePartsPrefixBlank() {
        MessagePartExtractorSettings settings = new MessagePartExtractorSettings();
        settings.m_imagePartsPrefix = Optional.of("\t"); // Set to blank

        InvalidSettingsException thrown = assertThrows(InvalidSettingsException.class, () -> {
            MessageCellSplitterFactories.createCellSplitterFactories(settings, 0, SPEC);
        });
        assertEquals("Image parts column prefix cannot be empty when 'Image parts column prefix' is enabled.", thrown.getMessage());
    }

    @Test
    void testCreateCellSplitterFactories_throwsOnToolCallsPrefixBlank() {
        MessagePartExtractorSettings settings = new MessagePartExtractorSettings();
        settings.m_toolCallsPrefix = Optional.of("\t"); // Set to blank

        InvalidSettingsException thrown = assertThrows(InvalidSettingsException.class, () -> {
            MessageCellSplitterFactories.createCellSplitterFactories(settings, 0, SPEC);
        });
        assertEquals("Tool calls column prefix cannot be empty when 'Tool calls column prefix' is enabled.", thrown.getMessage());
    }

    @Test
    void testCreateCellSplitterFactories_throwsOnToolCallIDPrefixBlank() {
        MessagePartExtractorSettings settings = new MessagePartExtractorSettings();
        settings.m_toolCallIdColumnName = Optional.of("\t"); // Set to blank

        InvalidSettingsException thrown = assertThrows(InvalidSettingsException.class, () -> {
            MessageCellSplitterFactories.createCellSplitterFactories(settings, 0, SPEC);
        });
        assertEquals("Tool call ID column name cannot be empty when 'Tool call ID column name' is enabled.", thrown.getMessage());
    }

    @Test
    void testCreateColumnSpecCreator() {
        var creator = MessageCellSplitterFactories.createColumnSpecCreator("prefix", StringCell.TYPE, SPEC);
        DataColumnSpec spec = creator.apply(0);
        assertEquals("prefix1", spec.getName());
        assertEquals(StringCell.TYPE, spec.getType());
    }
    @Test
    void testCreateContentCounter() {
        var msg = TestUtil.createMessageWithTextParts("foo", "bar");
        DataCell cell = msg;
        var counter = MessageCellSplitterFactories.createContentCounter(part -> part.getType() == MessageContentPartType.TEXT);
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

    @Test
    void testTextPartsSplitterFactory_execution_withDataRow() throws InvalidSettingsException {
        MessagePartExtractorSettings settings = new MessagePartExtractorSettings();
        settings.m_textPartsPrefix = Optional.of("text_");
        settings.m_imagePartsPrefix = Optional.empty();
        settings.m_toolCallsPrefix = Optional.empty();
        settings.m_roleColumnName = Optional.empty();
        settings.m_toolCallIdColumnName = Optional.empty();
        settings.m_nameColumnName = Optional.empty();
        var factories = MessageCellSplitterFactories.createCellSplitterFactories(settings, 0, SPEC);
        var factory = factories.get(0);
        factory.accept(TestUtil.createMessageWithTextParts("foo", "bar"));
        var cellFactory = factory.get();
        DataRow row = new DefaultRow(new RowKey("row1"), TestUtil.createMessageWithTextParts("foo", "bar"));
        var cells = cellFactory.getCells(row);
        assertEquals(2, cells.length);
        assertEquals("foo", ((StringCell)cells[0]).getStringValue());
        assertEquals("bar", ((StringCell)cells[1]).getStringValue());
    }

    @Test
    void testImagePartsSplitterFactory_execution_withDataRow() throws InvalidSettingsException {
        MessagePartExtractorSettings settings = new MessagePartExtractorSettings();
        settings.m_textPartsPrefix = Optional.empty();
        settings.m_imagePartsPrefix = Optional.of("img_");
        settings.m_toolCallsPrefix = Optional.empty();
        settings.m_roleColumnName = Optional.empty();
        settings.m_toolCallIdColumnName = Optional.empty();
        settings.m_nameColumnName = Optional.empty();
        var factories = MessageCellSplitterFactories.createCellSplitterFactories(settings, 0, SPEC);
        var factory = factories.get(0);
        factory.accept(TestUtil.createMessageWithTextParts("foo"));
        var cellFactory = factory.get();
        DataRow row = new DefaultRow(new RowKey("row2"), TestUtil.createMessageWithTextParts("foo"));
        var cells = cellFactory.getCells(row);
        assertEquals(0, cells.length); // No PNG parts
    }

    @Test
    void testToolCallsSplitterFactory_execution_withDataRow() throws InvalidSettingsException {
        MessagePartExtractorSettings settings = new MessagePartExtractorSettings();
        settings.m_textPartsPrefix = Optional.empty();
        settings.m_imagePartsPrefix = Optional.empty();
        settings.m_toolCallsPrefix = Optional.of("tool_");
        settings.m_roleColumnName = Optional.empty();
        settings.m_toolCallIdColumnName = Optional.empty();
        settings.m_nameColumnName = Optional.empty();
        var factories = MessageCellSplitterFactories.createCellSplitterFactories(settings, 0, SPEC);
        var factory = factories.get(0);
        factory.accept(TestUtil.createMessageWithToolCalls(2));
        var cellFactory = factory.get();
        DataRow row = new DefaultRow(new RowKey("row3"), TestUtil.createMessageWithToolCalls(2));
        var cells = cellFactory.getCells(row);
        assertEquals(2, cells.length);
        for (DataCell cell : cells) {
            assertEquals(JSONCellFactory.TYPE, cell.getType());
        }
    }

    @Test
    void testRoleSplitterFactory_execution_withDataRow() throws InvalidSettingsException {
        MessagePartExtractorSettings settings = new MessagePartExtractorSettings();
        settings.m_textPartsPrefix = Optional.empty();
        settings.m_imagePartsPrefix = Optional.empty();
        settings.m_toolCallsPrefix = Optional.empty();
        settings.m_roleColumnName = Optional.of("role");
        settings.m_toolCallIdColumnName = Optional.empty();
        settings.m_nameColumnName = Optional.empty();
        var factories = MessageCellSplitterFactories.createCellSplitterFactories(settings, 0, SPEC);
        var factory = factories.get(0);
        var cellFactory = factory.get();
        DataRow row = new DefaultRow(new RowKey("row4"), TestUtil.createMessageWithRole("USER"));
        var cells = cellFactory.getCells(row, 0);
        assertEquals(1, cells.length);
        assertEquals("USER", ((StringCell)cells[0]).getStringValue());
    }

    @Test
    void testToolCallIdSplitterFactory_execution_withDataRow() throws InvalidSettingsException {
        MessagePartExtractorSettings settings = new MessagePartExtractorSettings();
        settings.m_textPartsPrefix = Optional.empty();
        settings.m_imagePartsPrefix = Optional.empty();
        settings.m_toolCallsPrefix = Optional.empty();
        settings.m_roleColumnName = Optional.empty();
        settings.m_toolCallIdColumnName = Optional.of("toolCallId");
        settings.m_nameColumnName = Optional.empty();
        var factories = MessageCellSplitterFactories.createCellSplitterFactories(settings, 0, SPEC);
        var factory = factories.get(0);
        var cellFactory = factory.get();
        DataRow row = new DefaultRow(new RowKey("row5"), TestUtil.createMessageWithToolCallId("id123"));
        var cells = cellFactory.getCells(row, 0);
        assertEquals(1, cells.length);
        assertEquals("id123", ((StringCell)cells[0]).getStringValue());
    }

    @Test
    void testNameSplitterFactory_execution_withDataRow() throws InvalidSettingsException {
        MessagePartExtractorSettings settings = new MessagePartExtractorSettings();
        settings.m_textPartsPrefix = Optional.empty();
        settings.m_imagePartsPrefix = Optional.empty();
        settings.m_toolCallsPrefix = Optional.empty();
        settings.m_roleColumnName = Optional.empty();
        settings.m_toolCallIdColumnName = Optional.empty();
        settings.m_nameColumnName = Optional.of("name");
        var factories = MessageCellSplitterFactories.createCellSplitterFactories(settings, 0, SPEC);
        var factory = factories.get(0);
        var cellFactory = factory.get();
        DataRow row = new DefaultRow(new RowKey("row6"), TestUtil.createMessageWithName("foo"));
        var cells = cellFactory.getCells(row, 0);
        assertEquals(1, cells.length);
        assertEquals("foo", ((StringCell)cells[0]).getStringValue());
    }

    @Test
    void testTextPartsSplitterFactory_multiRowExtraction() throws InvalidSettingsException {
        MessagePartExtractorSettings settings = new MessagePartExtractorSettings();
        settings.m_textPartsPrefix = Optional.of("text_");
        settings.m_imagePartsPrefix = Optional.empty();
        settings.m_toolCallsPrefix = Optional.empty();
        settings.m_roleColumnName = Optional.empty();
        settings.m_toolCallIdColumnName = Optional.empty();
        settings.m_nameColumnName = Optional.empty();
        var factories = MessageCellSplitterFactories.createCellSplitterFactories(settings, 0, SPEC);
        var factory = factories.get(0);
        // Accept both rows to simulate state collection (max parts = 3)
        factory.accept(TestUtil.createMessageWithTextParts("a", "b"));
        factory.accept(TestUtil.createMessageWithTextParts("x", "y", "z"));
        var cellFactory = factory.get();
        DataRow row1 = new DefaultRow(new RowKey("row1"), TestUtil.createMessageWithTextParts("a", "b"));
        DataRow row2 = new DefaultRow(new RowKey("row2"), TestUtil.createMessageWithTextParts("x", "y", "z"));
        var cells1 = cellFactory.getCells(row1);
        var cells2 = cellFactory.getCells(row2);
        // Should pad row1 to 3 columns
        assertEquals(3, cells1.length);
        assertEquals("a", ((StringCell)cells1[0]).getStringValue());
        assertEquals("b", ((StringCell)cells1[1]).getStringValue());
        assertTrue(cells1[2].isMissing());
        // Row2 should have all 3 values
        assertEquals(3, cells2.length);
        assertEquals("x", ((StringCell)cells2[0]).getStringValue());
        assertEquals("y", ((StringCell)cells2[1]).getStringValue());
        assertEquals("z", ((StringCell)cells2[2]).getStringValue());
    }

    @Test
    void testToolCallsSplitterFactory_multiRowExtraction() throws InvalidSettingsException {
        MessagePartExtractorSettings settings = new MessagePartExtractorSettings();
        settings.m_textPartsPrefix = Optional.empty();
        settings.m_imagePartsPrefix = Optional.empty();
        settings.m_toolCallsPrefix = Optional.of("tool_");
        settings.m_roleColumnName = Optional.empty();
        settings.m_toolCallIdColumnName = Optional.empty();
        settings.m_nameColumnName = Optional.empty();
        var factories = MessageCellSplitterFactories.createCellSplitterFactories(settings, 0, SPEC);
        var factory = factories.get(0);
        // Accept both rows to simulate state collection (max tool calls = 3)
        factory.accept(TestUtil.createMessageWithToolCalls(2));
        factory.accept(TestUtil.createMessageWithToolCalls(3));
        var cellFactory = factory.get();
        DataRow row1 = new DefaultRow(new RowKey("row1"), TestUtil.createMessageWithToolCalls(2));
        DataRow row2 = new DefaultRow(new RowKey("row2"), TestUtil.createMessageWithToolCalls(3));
        var cells1 = cellFactory.getCells(row1);
        var cells2 = cellFactory.getCells(row2);
        // Should pad row1 to 3 columns
        assertEquals(3, cells1.length);
        assertEquals(JSONCellFactory.TYPE, cells1[0].getType());
        assertEquals(JSONCellFactory.TYPE, cells1[1].getType());
        assertTrue(cells1[2].isMissing());
        // Row2 should have all 3 tool calls
        assertEquals(3, cells2.length);
        assertEquals(JSONCellFactory.TYPE, cells2[0].getType());
        assertEquals(JSONCellFactory.TYPE, cells2[1].getType());
        assertEquals(JSONCellFactory.TYPE, cells2[2].getType());
    }

    @Test
    void testCreateColumnSpecCreator_withDuplicateColumnName() {
        var creator = MessageCellSplitterFactories.createColumnSpecCreator("text", StringCell.TYPE, SPEC);

        DataColumnSpec spec1 = creator.apply(0);
        DataColumnSpec spec2 = creator.apply(1);

        assertNotNull(spec1);
        assertNotNull(spec2);
        assertNotEquals(spec1.getName(), spec2.getName());
        assertEquals("text1 (#1)", spec1.getName());
        assertEquals("text2", spec2.getName());
    }

    @Test
    void testRoleSplitterFactory_execution_withDuplicateRoleColumnName() throws InvalidSettingsException {
        MessagePartExtractorSettings settings = new MessagePartExtractorSettings();
        settings.m_textPartsPrefix = Optional.empty();
        settings.m_imagePartsPrefix = Optional.empty();
        settings.m_toolCallsPrefix = Optional.empty();
        settings.m_roleColumnName = Optional.of("role");
        settings.m_toolCallIdColumnName = Optional.empty();
        settings.m_nameColumnName = Optional.empty();

        var factories = MessageCellSplitterFactories.createCellSplitterFactories(settings, 0, SPEC);
        var factory = factories.get(0);
        var cellFactory = factory.get();

        DataRow row = new DefaultRow(new RowKey("row1"), TestUtil.createMessageWithRole("USER"));
        var cells = cellFactory.getCells(row, 0);

        assertEquals(1, cells.length);
        assertEquals("USER", ((StringCell)cells[0]).getStringValue());
        assertEquals("role (#1)", cellFactory.getColumnSpecs()[0].getName());
    }
}
