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
 *   May 30, 2025 (Seray Arslan, KNIME GmbH, Konstanz, Germany): created
 */
package org.knime.ai.core.node.message.create;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

import org.eclipse.core.runtime.FileLocator;
import org.junit.jupiter.api.Test;
import org.knime.ai.core.data.message.MessageCell;
import org.knime.ai.core.data.message.MessageValue.MessageContentPart;
import org.knime.ai.core.data.message.MessageValue.MessageType;
import org.knime.ai.core.data.message.MessageValue.ToolCall;
import org.knime.ai.core.data.message.TextContentPart;
import org.knime.core.data.DataCell;
import org.knime.core.data.DataColumnSpec;
import org.knime.core.data.DataColumnSpecCreator;
import org.knime.core.data.DataTableSpec;
import org.knime.core.data.DataType;
import org.knime.core.data.def.DefaultRow;
import org.knime.core.data.def.StringCell;
import org.knime.core.data.image.png.PNGImageCellFactory;
import org.knime.core.util.FileUtil;
import org.knime.core.webui.node.dialog.defaultdialog.setting.singleselection.StringOrEnum;
import org.osgi.framework.FrameworkUtil;

@SuppressWarnings({"restriction", "static-method"})
final class MessageCellCreatorTest {

    private static MessageCreatorNodeSettings minimalSettings() {
        MessageCreatorNodeSettings settings = new MessageCreatorNodeSettings();
        settings.m_roleInputType = MessageCreatorNodeSettings.InputType.VALUE;
        settings.m_roleValue = MessageType.USER;
        settings.m_content = new MessageCreatorNodeSettings.Contents[] {
            new MessageCreatorNodeSettings.Contents("Hello World", "")
        };
        settings.m_toolCalls = new MessageCreatorNodeSettings.ToolCallSettings[]{};
        settings.m_messageColumnName = "Message";
        return settings;
    }

    private static DataTableSpec minimalSpec() {
        return new DataTableSpec();
    }

    @Test
    void testCreateTextContentExtractor_value() {
        var settings = minimalSettings();
        var spec = minimalSpec();
        var creator = new MessageCellCreator(settings, spec);
        var extractor = creator.createTextContentExtractor(settings.m_content[0]);
        var row = new DefaultRow("row0", List.of());
        Optional<MessageContentPart> part = extractor.apply(row);
        assertTrue(part.isPresent());
        assertEquals("Hello World", ((TextContentPart)part.get()).getContent());
    }

    @Test
    void testCreateTextContentExtractor_column() {
        var settings = minimalSettings();
        settings.m_content = new MessageCreatorNodeSettings.Contents[] {
            new MessageCreatorNodeSettings.Contents()
        };
        settings.m_content[0].m_contentType = MessageCreatorNodeSettings.Contents.ContentType.TEXT;
        settings.m_content[0].m_inputType = MessageCreatorNodeSettings.InputType.COLUMN;
        settings.m_content[0].m_textColumn = "textCol";
        var spec = new DataTableSpec(
            new DataColumnSpecCreator("textCol", StringCell.TYPE).createSpec()
        );
        var creator = new MessageCellCreator(settings, spec);
        var extractor = creator.createTextContentExtractor(settings.m_content[0]);
        var row = new DefaultRow("row0", new StringCell("column value"));
        var part = extractor.apply(row);
        assertTrue(part.isPresent());
        assertEquals("column value", ((TextContentPart)part.get()).getContent());
    }

    @Test
    void testCreateTextContentExtractor_column_missingValue() {
        var settings = minimalSettings();
        settings.m_content = new MessageCreatorNodeSettings.Contents[] {
            new MessageCreatorNodeSettings.Contents()
        };
        settings.m_content[0].m_contentType = MessageCreatorNodeSettings.Contents.ContentType.TEXT;
        settings.m_content[0].m_inputType = MessageCreatorNodeSettings.InputType.COLUMN;
        settings.m_content[0].m_textColumn = "textCol";
        var spec = new DataTableSpec(
            new DataColumnSpecCreator("textCol", StringCell.TYPE).createSpec()
        );
        var creator = new MessageCellCreator(settings, spec);
        var extractor = creator.createTextContentExtractor(settings.m_content[0]);
        var row = new DefaultRow("row0", DataType.getMissingCell());
        var part = extractor.apply(row);
        assertTrue(part.isEmpty());
    }

    @Test
    void testCreateContentExtractor_value() {
        var settings = minimalSettings();
        var spec = minimalSpec();
        var creator = new MessageCellCreator(settings, spec);
        var extractor = creator.createContentExtractor();
        var row = new DefaultRow("row0", List.of());
        List<MessageContentPart> parts = extractor.apply(row);
        assertEquals(1, parts.size());
        assertEquals("Hello World", ((TextContentPart)parts.get(0)).getContent());
    }

    @Test
    void testCreateRoleExtractor_value() {
        var settings = minimalSettings();
        var spec = minimalSpec();
        var creator = new MessageCellCreator(settings, spec);
        var extractor = creator.createRoleExtractor();
        var row = new DefaultRow("row0", List.of());
        assertEquals(MessageType.USER, extractor.apply(row));
    }

    @Test
    void testCreateRoleExtractor_column() {
        var settings = minimalSettings();
        settings.m_roleInputType = MessageCreatorNodeSettings.InputType.COLUMN;
        settings.m_roleColumn = "role";
        var spec = new DataTableSpec(
            new DataColumnSpecCreator("role", StringCell.TYPE).createSpec()
        );
        var creator = new MessageCellCreator(settings, spec);
        var extractor = creator.createRoleExtractor();
        var row = new DefaultRow("row0", new StringCell("user"));
        assertEquals(MessageType.USER, extractor.apply(row));
    }

    @Test
    void testCreateRoleExtractor_column_missing() {
        var settings = minimalSettings();
        settings.m_roleInputType = MessageCreatorNodeSettings.InputType.COLUMN;
        settings.m_roleColumn = "role";
        var spec = new DataTableSpec(
            new DataColumnSpecCreator("role", StringCell.TYPE).createSpec()
        );
        var creator = new MessageCellCreator(settings, spec);
        var extractor = creator.createRoleExtractor();
        var row = new DefaultRow("row0", DataType.getMissingCell());
        try {
            extractor.apply(row);
            fail("Expected IllegalArgumentException for missing role");
        } catch (IllegalArgumentException e) {
            // expected
        }
    }

    @Test
    void testCreateMessageCellCreator_value() {
        var settings = minimalSettings();
        var spec = minimalSpec();
        var creator = new MessageCellCreator(settings, spec);
        var cellCreator = creator.createMessageCellCreator();
        var row = new DefaultRow("row0", List.of());
        DataCell cell = cellCreator.apply(row);
        assertTrue(cell instanceof MessageCell);
        MessageCell msgCell = (MessageCell)cell;
        assertEquals(MessageType.USER, msgCell.getMessageType());
        assertEquals(1, msgCell.getContent().size());
        assertEquals("Hello World", ((TextContentPart)msgCell.getContent().get(0)).getContent());
        assertTrue(msgCell.getToolCalls().isEmpty());
        assertTrue(msgCell.getToolCallId().isEmpty());
    }

    private static DataColumnSpec createColumnSpec(final String name, final DataType type) {
        return new DataColumnSpecCreator(name, type).createSpec();
    }

    @Test
    void testCreateToolCallExtractor_missingCells() {
        var settings = minimalSettings();
        var tc = new MessageCreatorNodeSettings.ToolCallSettings();
        tc.m_toolNameColumn = "toolName";
        tc.m_toolIdColumn = "toolId";
        tc.m_argumentsColumn = "args";
        settings.m_toolCalls = new MessageCreatorNodeSettings.ToolCallSettings[]{tc};
        var spec = new DataTableSpec(
            createColumnSpec("toolName", StringCell.TYPE),
            createColumnSpec("toolId", StringCell.TYPE),
            createColumnSpec("args", StringCell.TYPE)
        );
        var creator = new MessageCellCreator(settings, spec);
        var extractor = creator.createToolCallExtractor(tc);
        // All cells missing
        var row = new DefaultRow("row0", DataType.getMissingCell(), DataType.getMissingCell(), DataType.getMissingCell());
        assertTrue(extractor.apply(row).isEmpty());
    }

    @Test
    void testCreateToolCallExtractor_present() {
        var settings = minimalSettings();
        var tc = new MessageCreatorNodeSettings.ToolCallSettings();
        tc.m_toolNameColumn = "toolName";
        tc.m_toolIdColumn = "toolId";
        tc.m_argumentsColumn = "args";
        settings.m_toolCalls = new MessageCreatorNodeSettings.ToolCallSettings[]{tc};
        var spec = new DataTableSpec(
            createColumnSpec("toolName", StringCell.TYPE),
            createColumnSpec("toolId", StringCell.TYPE),
            createColumnSpec("args", StringCell.TYPE)
        );
        var creator = new MessageCellCreator(settings, spec);
        var extractor = creator.createToolCallExtractor(tc);
        var row = new DefaultRow("row0", new StringCell("tool"), new StringCell("id"), new StringCell("{}"));
        Optional<ToolCall> call = extractor.apply(row);
        assertTrue(call.isPresent());
        assertEquals("tool", call.get().toolName());
        assertEquals("id", call.get().id());
        assertEquals("{}", call.get().arguments());
    }

    @Test
    void testCreateToolCallIdExtractor_column() {
        var settings = minimalSettings();
        settings.m_toolCallIdColumn = new StringOrEnum<>("toolCallId");
        var spec = new DataTableSpec(
            new DataColumnSpecCreator("toolCallId", StringCell.TYPE).createSpec()
        );
        var creator = new MessageCellCreator(settings, spec);
        var extractor = creator.createToolCallIdExtractor();
        var row = new DefaultRow("row0", new StringCell("id-123"));
        assertEquals("id-123", extractor.apply(row));
    }

    @Test
    void testCreateToolCallIdExtractor_column_missing() {
        var settings = minimalSettings();
        settings.m_toolCallIdColumn = new StringOrEnum<>("toolCallId");
        var spec = new DataTableSpec(
            new DataColumnSpecCreator("toolCallId", StringCell.TYPE).createSpec()
        );
        var creator = new MessageCellCreator(settings, spec);
        var extractor = creator.createToolCallIdExtractor();
        var row = new DefaultRow("row0", DataType.getMissingCell());
        assertEquals(null, extractor.apply(row));
    }

    @Test
    void testCreateImageContentExtractor_column() throws IOException {
        var settings = minimalSettings();
        settings.m_content = new MessageCreatorNodeSettings.Contents[] {
            new MessageCreatorNodeSettings.Contents()
        };
        settings.m_content[0].m_contentType = MessageCreatorNodeSettings.Contents.ContentType.IMAGE;
        settings.m_content[0].m_imageColumn = "imgCol";
        var spec = new DataTableSpec(
            new DataColumnSpecCreator("imgCol", PNGImageCellFactory.TYPE).createSpec()
        );
        var creator = new MessageCellCreator(settings, spec);
        var extractor = creator.createImageContentExtractor(settings.m_content[0]);

        var imgFile = getFileResource("Message-creator.png");
        var imgBytes = Files.readAllBytes(imgFile);
        var imgCell = PNGImageCellFactory.create(imgBytes);
        var row = new DefaultRow("row0", imgCell);
        var part = extractor.apply(row);
        assertTrue(part.isPresent());
        assertArrayEquals(imgBytes, part.get().getData());
    }

    @Test
    void testCreateImageContentExtractor_missing() {
        var settings = minimalSettings();
        settings.m_content = new MessageCreatorNodeSettings.Contents[] {
            new MessageCreatorNodeSettings.Contents()
        };
        settings.m_content[0].m_contentType = MessageCreatorNodeSettings.Contents.ContentType.IMAGE;
        settings.m_content[0].m_imageColumn = "imgCol";
        var spec = new DataTableSpec(
            new DataColumnSpecCreator("imgCol", PNGImageCellFactory.TYPE).createSpec()
        );
        var creator = new MessageCellCreator(settings, spec);
        var extractor = creator.createImageContentExtractor(settings.m_content[0]);
        var row = new DefaultRow("row0", DataType.getMissingCell());
        var part = extractor.apply(row);
        assertTrue(part.isEmpty());
    }

    @Test
    void testCreateNameExtractor_noColumnSelected() {
        var settings = minimalSettings();
        // m_nameColumn is left as default (NoneChoice)
        var spec = minimalSpec();
        var creator = new MessageCellCreator(settings, spec);
        var extractor = creator.createNameExtractor();
        var row = new DefaultRow("row0", List.of());
        assertTrue(extractor.apply(row).isEmpty());
    }

    @Test
    void testCreateNameExtractor_missingValue() {
        var settings = minimalSettings();
        settings.m_nameColumn = new StringOrEnum<>("name");
        var spec = new DataTableSpec(
            new DataColumnSpecCreator("name", StringCell.TYPE).createSpec()
        );
        var creator = new MessageCellCreator(settings, spec);
        var extractor = creator.createNameExtractor();
        var row = new DefaultRow("row0", DataType.getMissingCell());
        assertTrue(extractor.apply(row).isEmpty());
    }

    @Test
    void testCreateNameExtractor_presentValue() {
        var settings = minimalSettings();
        settings.m_nameColumn = new StringOrEnum<>("name");
        var spec = new DataTableSpec(
            new DataColumnSpecCreator("name", StringCell.TYPE).createSpec()
        );
        var creator = new MessageCellCreator(settings, spec);
        var extractor = creator.createNameExtractor();
        var row = new DefaultRow("row0", new StringCell("Alice"));
        assertTrue(extractor.apply(row).isPresent());
        assertEquals("Alice", extractor.apply(row).get());
    }

    private static Path getFileResource(final String fileName) throws IOException {
        var url = FileLocator.toFileURL(resolveToURL("/files/" + fileName, MessageCellCreatorTest.class));
        return FileUtil.getFileFromURL(url).toPath();
    }

    /**
     * @param path defines the target location
     * @param clazz defines the bundle to use as root
     * @return the location as file URL
     * @throws IOException
     */
    private static URL resolveToURL(final String path, final Class<?> clazz) throws IOException {
        var bundle = FrameworkUtil.getBundle(clazz);
        var p = new org.eclipse.core.runtime.Path(path);
        var url = FileLocator.find(bundle, p, null);
        if (url == null) {
            throw new FileNotFoundException("Path " + path + " does not exist in bundle " + bundle.getSymbolicName());
        }
        return url;
    }
}
