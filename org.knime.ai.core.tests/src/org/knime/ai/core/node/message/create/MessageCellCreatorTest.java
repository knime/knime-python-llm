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
import static org.junit.jupiter.api.Assertions.assertThrows;
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
import org.knime.ai.core.data.message.PngContentPart;
import org.knime.ai.core.data.message.TextContentPart;
import org.knime.ai.core.node.message.create.MessageCreatorNodeSettings.InputType;
import org.knime.core.data.DataCell;
import org.knime.core.data.DataColumnSpec;
import org.knime.core.data.DataColumnSpecCreator;
import org.knime.core.data.DataTableSpec;
import org.knime.core.data.DataType;
import org.knime.core.data.def.DefaultRow;
import org.knime.core.data.def.StringCell;
import org.knime.core.data.image.png.PNGImageCellFactory;
import org.knime.core.node.InvalidSettingsException;
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
    void testCreateTextContentExtractor_value() throws InvalidSettingsException {
        var settings = minimalSettings();
        var spec = minimalSpec();
        var creator = new MessageCellCreator(settings, spec);
        var extractor = creator.createTextContentExtractor(settings.m_content[0], 0);
        var row = new DefaultRow("row0", List.of());
        Optional<MessageContentPart> part = extractor.apply(row);
        assertTrue(part.isPresent());
        assertEquals("Hello World", ((TextContentPart)part.get()).getContent());
    }

    @Test
    void testCreateTextContentExtractor_column() throws InvalidSettingsException {
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
        var extractor = creator.createTextContentExtractor(settings.m_content[0], 0);
        var row = new DefaultRow("row0", new StringCell("column value"));
        var part = extractor.apply(row);
        assertTrue(part.isPresent());
        assertEquals("column value", ((TextContentPart)part.get()).getContent());
    }

    @Test
    void testCreateTextContentExtractor_column_missingValue() throws InvalidSettingsException {
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
        var extractor = creator.createTextContentExtractor(settings.m_content[0], 0);
        var row = new DefaultRow("row0", DataType.getMissingCell());
        var part = extractor.apply(row);
        assertTrue(part.isEmpty());
    }

    @Test
    void testCreateContentExtractor_value() throws InvalidSettingsException {
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
    void testCreateRoleExtractor_value() throws InvalidSettingsException {
        var settings = minimalSettings();
        var spec = minimalSpec();
        var creator = new MessageCellCreator(settings, spec);
        var extractor = creator.createRoleExtractor();
        var row = new DefaultRow("row0", List.of());
        assertEquals(MessageType.USER, extractor.apply(row));
    }

    @Test
    void testCreateRoleExtractor_column() throws InvalidSettingsException {
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
    void testCreateRoleExtractor_column_missing() throws InvalidSettingsException {
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
    void testCreateRoleExtractor_column_invalidValue() throws InvalidSettingsException{
        var settings = minimalSettings();
        settings.m_roleInputType = MessageCreatorNodeSettings.InputType.COLUMN;
        settings.m_roleColumn = "role";
        var spec = new DataTableSpec(
            new DataColumnSpecCreator("role", StringCell.TYPE).createSpec()
        );
        var creator = new MessageCellCreator(settings, spec);
        var extractor = creator.createRoleExtractor();
        var row = new DefaultRow("row0", new StringCell("notarole"));
        try {
            extractor.apply(row);
            fail("Expected IllegalArgumentException for invalid role value");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("invalid value"));
            assertTrue(e.getMessage().contains("Valid roles"));
        }
    }

    @Test
    void testCreateMessageCellCreator_value() throws InvalidSettingsException {
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
    void testCreateToolCallExtractor_missingCells() throws InvalidSettingsException {
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
        var extractor = creator.createToolCallExtractor(tc, 0);
        // All cells missing
        var row = new DefaultRow("row0", DataType.getMissingCell(), DataType.getMissingCell(), DataType.getMissingCell());
        assertTrue(extractor.apply(row).isEmpty());
    }

    @Test
    void testCreateToolCallExtractor_present() throws InvalidSettingsException {
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
        var extractor = creator.createToolCallExtractor(tc, 0);
        var row = new DefaultRow("row0", new StringCell("tool"), new StringCell("id"), new StringCell("{}"));
        Optional<ToolCall> call = extractor.apply(row);
        assertTrue(call.isPresent());
        assertEquals("tool", call.get().toolName());
        assertEquals("id", call.get().id());
        assertEquals("{}", call.get().arguments());
    }

    @Test
    void testCreateToolCallsExtractor() throws InvalidSettingsException {
        var settings = minimalSettings();

        var tc1 = new MessageCreatorNodeSettings.ToolCallSettings();
        tc1.m_toolNameColumn = "toolName1";
        tc1.m_toolIdColumn = "toolId1";
        tc1.m_argumentsColumn = "args1";

        var tc2 = new MessageCreatorNodeSettings.ToolCallSettings();
        tc2.m_toolNameColumn = "toolName2";
        tc2.m_toolIdColumn = "toolId2";
        tc2.m_argumentsColumn = "args2";

        settings.m_toolCalls = new MessageCreatorNodeSettings.ToolCallSettings[]{tc1, tc2};

        var spec = new DataTableSpec(
            createColumnSpec("toolName1", StringCell.TYPE),
            createColumnSpec("toolId1", StringCell.TYPE),
            createColumnSpec("args1", StringCell.TYPE),
            createColumnSpec("toolName2", StringCell.TYPE),
            createColumnSpec("toolId2", StringCell.TYPE),
            createColumnSpec("args2", StringCell.TYPE)
        );

        var creator = new MessageCellCreator(settings, spec);
        var extractor = creator.createToolCallsExtractor();

        var row = new DefaultRow("row0",
            new StringCell("tool1"), new StringCell("id1"), new StringCell("{}"),
            new StringCell("tool2"), new StringCell("id2"), new StringCell("[]")
        );

        List<ToolCall> toolCalls = extractor.apply(row);

        assertEquals(2, toolCalls.size());

        assertEquals("tool1", toolCalls.get(0).toolName());
        assertEquals("id1", toolCalls.get(0).id());
        assertEquals("{}", toolCalls.get(0).arguments());

        assertEquals("tool2", toolCalls.get(1).toolName());
        assertEquals("id2", toolCalls.get(1).id());
        assertEquals("[]", toolCalls.get(1).arguments());
    }

    @Test
    void testCreateToolCallIdExtractor_column() throws InvalidSettingsException {
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
    void testCreateToolCallIdExtractor_column_missing() throws InvalidSettingsException {
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
    void testCreateImageContentExtractor_column() throws IOException, InvalidSettingsException {
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
        var extractor = creator.createImageContentExtractor(settings.m_content[0], 0);

        var imgFile = getFileResource("Message-creator.png");
        var imgBytes = Files.readAllBytes(imgFile);
        var imgCell = PNGImageCellFactory.create(imgBytes);
        var row = new DefaultRow("row0", imgCell);
        var part = extractor.apply(row);
        assertTrue(part.isPresent());
        assertArrayEquals(imgBytes, part.get().getData());
    }

    @Test
    void testCreateImageContentExtractor_missing() throws InvalidSettingsException {
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
        var extractor = creator.createImageContentExtractor(settings.m_content[0], 0);
        var row = new DefaultRow("row0", DataType.getMissingCell());
        var part = extractor.apply(row);
        assertTrue(part.isEmpty());
    }

    @Test
    void testCreateNameExtractor_noColumnSelected() throws InvalidSettingsException {
        var settings = minimalSettings();
        // m_nameColumn is left as default (NoneChoice)
        var spec = minimalSpec();
        var creator = new MessageCellCreator(settings, spec);
        var extractor = creator.createNameExtractor();
        var row = new DefaultRow("row0", List.of());
        assertTrue(extractor.apply(row).isEmpty());
    }

    @Test
    void testCreateNameExtractor_missingValue() throws InvalidSettingsException {
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
    void testCreateNameExtractor_presentValue() throws InvalidSettingsException {
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

    @Test
    void testCreateRoleExtractor_column_noColumnSelected() {
        var settings = minimalSettings();
        settings.m_roleInputType = MessageCreatorNodeSettings.InputType.COLUMN;
        settings.m_roleColumn = null;

        var spec = new DataTableSpec();

        var exception = assertThrows(InvalidSettingsException.class, () -> {
            new MessageCellCreator(settings, spec).createRoleExtractor();
        });

        assertEquals("Please select a valid column for the Role column.", exception.getMessage());
    }

    @Test
    void testCreateRoleExtractor_column_nonExistentColumn() {
        var settings = minimalSettings();
        settings.m_roleInputType = MessageCreatorNodeSettings.InputType.COLUMN;
        settings.m_roleColumn = "ColumnThatDoesNotExist";

        var spec = new DataTableSpec();

        var exception = assertThrows(InvalidSettingsException.class, () -> {
            new MessageCellCreator(settings, spec).createRoleExtractor();
        });

        assertEquals("The selected column 'ColumnThatDoesNotExist' is not part of the input table.", exception.getMessage());
    }


    @Test
    void testCreateTextContentExtractor_column_nullColumnSelected() {
        var settings = minimalSettings();
        settings.m_content = new MessageCreatorNodeSettings.Contents[] {
            new MessageCreatorNodeSettings.Contents()
        };
        settings.m_content[0].m_contentType = MessageCreatorNodeSettings.Contents.ContentType.TEXT;
        settings.m_content[0].m_inputType = InputType.COLUMN;
        settings.m_content[0].m_textColumn = null;

        var spec = new DataTableSpec();

        var exception = assertThrows(InvalidSettingsException.class, () -> {
            new MessageCellCreator(settings, spec).createTextContentExtractor(settings.m_content[0], 0);
        });

        assertEquals("Please select a valid column for the Text column in Content 0.", exception.getMessage());
    }

    @Test
    void testCreateTextContentExtractor_column_nonExistentColumn() {
        var settings = minimalSettings();
        settings.m_content = new MessageCreatorNodeSettings.Contents[] {
            new MessageCreatorNodeSettings.Contents()
        };
        settings.m_content[0].m_contentType = MessageCreatorNodeSettings.Contents.ContentType.TEXT;
        settings.m_content[0].m_inputType = InputType.COLUMN;
        settings.m_content[0].m_textColumn = "NonExistentTextColumn";

        var spec = new DataTableSpec();

        var exception = assertThrows(InvalidSettingsException.class, () -> {
            new MessageCellCreator(settings, spec).createTextContentExtractor(settings.m_content[0], 0);
        });

        assertEquals("The selected column 'NonExistentTextColumn' is not part of the input table.", exception.getMessage());
    }


    @Test
    void testCreateImageContentExtractor_column_nullColumnSelected() {
        var settings = minimalSettings();
        settings.m_content = new MessageCreatorNodeSettings.Contents[] {
            new MessageCreatorNodeSettings.Contents()
        };
        settings.m_content[0].m_contentType = MessageCreatorNodeSettings.Contents.ContentType.IMAGE;
        settings.m_content[0].m_imageColumn = null;

        var spec = new DataTableSpec();

        var exception = assertThrows(InvalidSettingsException.class, () -> {
            new MessageCellCreator(settings, spec).createImageContentExtractor(settings.m_content[0], 0);
        });

        assertEquals("Please select a valid column for the Image column in Content 0.", exception.getMessage());
    }

    @Test
    void testCreateImageContentExtractor_column_nonExistentColumn() {
        var settings = minimalSettings();
        settings.m_content = new MessageCreatorNodeSettings.Contents[] {
            new MessageCreatorNodeSettings.Contents()
        };
        settings.m_content[0].m_contentType = MessageCreatorNodeSettings.Contents.ContentType.IMAGE;
        settings.m_content[0].m_imageColumn = "NonExistentImageColumn";

        var spec = new DataTableSpec();

        var exception = assertThrows(InvalidSettingsException.class, () -> {
            new MessageCellCreator(settings, spec).createImageContentExtractor(settings.m_content[0], 0);
        });

        assertEquals("The selected column 'NonExistentImageColumn' is not part of the input table.", exception.getMessage());
    }

    @Test
    void testCreateToolCallExtractor_toolNameColumnNotSelected() {
        var settings = minimalSettings();

        settings.m_toolCalls = new MessageCreatorNodeSettings.ToolCallSettings[] {
            new MessageCreatorNodeSettings.ToolCallSettings()
        };
        settings.m_toolCalls[0].m_toolNameColumn = null;
        settings.m_toolCalls[0].m_toolIdColumn = "toolId";

        var spec = new DataTableSpec(
            new DataColumnSpecCreator("toolId", StringCell.TYPE).createSpec()
        );

        var exception = assertThrows(InvalidSettingsException.class, () -> {
            new MessageCellCreator(settings, spec).createToolCallExtractor(settings.m_toolCalls[0], 0);
        });

        assertEquals("Please select a valid column for the Tool name column in Tool Call 0.", exception.getMessage());
    }

    @Test
    void testCreateToolCallExtractor_toolIdColumnNotSelected(){
        var settings = minimalSettings();

        settings.m_toolCalls = new MessageCreatorNodeSettings.ToolCallSettings[] {
            new MessageCreatorNodeSettings.ToolCallSettings()
        };
        settings.m_toolCalls[0].m_toolNameColumn = "toolName";
        settings.m_toolCalls[0].m_toolIdColumn = null;

        var spec = new DataTableSpec(
            new DataColumnSpecCreator("toolName", StringCell.TYPE).createSpec()
        );
        var exception = assertThrows(InvalidSettingsException.class, () -> {
            new MessageCellCreator(settings, spec).createToolCallExtractor(settings.m_toolCalls[0], 0);
        });

        assertEquals("Please select a valid column for the Tool ID column in Tool Call 0.", exception.getMessage());
    }

    @Test
    void testCreateToolCallExtractor_argsColumnNotSelected() {
        var settings = minimalSettings();

        settings.m_toolCalls = new MessageCreatorNodeSettings.ToolCallSettings[] {
            new MessageCreatorNodeSettings.ToolCallSettings()
        };
        settings.m_toolCalls[0].m_toolNameColumn = "toolName";
        settings.m_toolCalls[0].m_toolIdColumn = "toolID";
        settings.m_toolCalls[0].m_argumentsColumn = null;

        var spec = new DataTableSpec(
            new DataColumnSpecCreator("toolName", StringCell.TYPE).createSpec(),
            new DataColumnSpecCreator("toolID", StringCell.TYPE).createSpec()
        );

        var exception = assertThrows(InvalidSettingsException.class, () -> {
            new MessageCellCreator(settings, spec).createToolCallExtractor(settings.m_toolCalls[0], 0);
        });

        assertEquals("Please select a valid column for the Arguments column in Tool Call 0.", exception.getMessage());
    }

    @Test
    void testCreateContentPartExtractor_mixedContent() throws InvalidSettingsException, IOException {
        var settings = minimalSettings();

        var textContent = new MessageCreatorNodeSettings.Contents();
        textContent.m_contentType = MessageCreatorNodeSettings.Contents.ContentType.TEXT;
        textContent.m_inputType = MessageCreatorNodeSettings.InputType.COLUMN;
        textContent.m_textColumn = "textCol";

        var imageContent = new MessageCreatorNodeSettings.Contents();
        imageContent.m_contentType = MessageCreatorNodeSettings.Contents.ContentType.IMAGE;
        imageContent.m_imageColumn = "imageCol";

        settings.m_content = new MessageCreatorNodeSettings.Contents[]{textContent, imageContent};

        var spec = new DataTableSpec(
            createColumnSpec("textCol", StringCell.TYPE),
            createColumnSpec("imageCol", PNGImageCellFactory.TYPE)
        );

        var creator = new MessageCellCreator(settings, spec);
        var contentExtractor = creator.createContentExtractor();

        var imgFile = getFileResource("Message-creator.png");
        var imgBytes = Files.readAllBytes(imgFile);
        var imgCell = PNGImageCellFactory.create(imgBytes);

        var row = new DefaultRow("row0", new StringCell("Sample Text"), imgCell);

        var parts = contentExtractor.apply(row);

        assertEquals(2, parts.size());

        assertTrue(parts.get(0) instanceof TextContentPart);
        assertEquals("Sample Text", ((TextContentPart)parts.get(0)).getContent());

        assertTrue(parts.get(1) instanceof PngContentPart);
        assertArrayEquals(imgBytes, ((PngContentPart)parts.get(1)).getData());
    }

    @Test
    void testCreateContentPartExtractor_unknownContentType() {
        var settings = minimalSettings();

        var unknownContent = new MessageCreatorNodeSettings.Contents();
        unknownContent.m_contentType = null;

        settings.m_content = new MessageCreatorNodeSettings.Contents[]{unknownContent};

        var spec = minimalSpec();
        var creator = new MessageCellCreator(settings, spec);

        var exception = assertThrows(IllegalArgumentException.class, () -> {
            creator.createContentPartExtractor(unknownContent, 1);
        });

        assertEquals("Unknown content type: null", exception.getMessage());
    }

}
