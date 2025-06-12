package org.knime.ai.core.data.message;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.Optional;

import org.junit.jupiter.api.Test;
import org.knime.ai.core.data.message.MessageValue.MessageContentPart;
import org.knime.ai.core.data.message.MessageValue.MessageType;
import org.knime.ai.core.data.message.MessageValue.ToolCall;

class MessageCellTest {

    @Test
    void testCreateUserMessageCell_basic() {
        TextContentPart text = new TextContentPart("Hello");
        MessageCell cell = MessageCell.createUserMessageCell(List.of(text));
        assertEquals(MessageType.USER, cell.getMessageType());
        assertEquals(1, cell.getContent().size());
        assertEquals(text, cell.getContent().get(0));
        assertTrue(cell.getToolCalls().isEmpty());
        assertTrue(cell.getToolCallId().isEmpty());
        assertTrue(cell.getName().isEmpty());
    }

    @Test
    void testCreateAIMessageCell_basic() {
        TextContentPart text = new TextContentPart("AI response");
        MessageCell cell = MessageCell.createAIMessageCell(List.of(text));
        assertEquals(MessageType.AI, cell.getMessageType());
        assertEquals(1, cell.getContent().size());
        assertEquals(text, cell.getContent().get(0));
        assertTrue(cell.getToolCalls().isEmpty());
        assertTrue(cell.getToolCallId().isEmpty());
        assertTrue(cell.getName().isEmpty());
    }

    @Test
    void testCreateAIMessageCell_withToolCalls() {
        TextContentPart text = new TextContentPart("AI with tool");
        ToolCall call = new ToolCall("tool", "id1", "{}{}");
        MessageCell cell = MessageCell.createAIMessageCell(List.of(text), List.of(call));
        assertEquals(MessageType.AI, cell.getMessageType());
        assertEquals(1, cell.getContent().size());
        assertEquals(text, cell.getContent().get(0));
        assertTrue(cell.getToolCalls().isPresent());
        assertEquals(1, cell.getToolCalls().get().size());
        assertEquals(call, cell.getToolCalls().get().get(0));
        assertTrue(cell.getToolCallId().isEmpty());
        assertTrue(cell.getName().isEmpty());
    }

    @Test
    void testCreateToolMessageCell_basic() {
        TextContentPart text = new TextContentPart("Tool response");
        MessageCell cell = MessageCell.createToolMessageCell(List.of(text), "callid", "toolname");
        assertEquals(MessageType.TOOL, cell.getMessageType());
        assertEquals(1, cell.getContent().size());
        assertEquals(text, cell.getContent().get(0));
        assertTrue(cell.getToolCalls().isEmpty());
        assertEquals(Optional.of("callid"), cell.getToolCallId());
        assertEquals(Optional.of("toolname"), cell.getName());
    }

    @Test
    void testEqualsAndHashCode() {
        TextContentPart text1 = new TextContentPart("A");
        TextContentPart text2 = new TextContentPart("A");
        MessageCell cell1 = MessageCell.createUserMessageCell(List.of(text1));
        MessageCell cell2 = MessageCell.createUserMessageCell(List.of(text2));
        assertEquals(cell1, cell2);
        assertEquals(cell1.hashCode(), cell2.hashCode());
    }

    @Test
    void testInequality() {
        TextContentPart text1 = new TextContentPart("A");
        TextContentPart text2 = new TextContentPart("B");
        MessageCell cell1 = MessageCell.createUserMessageCell(List.of(text1));
        MessageCell cell2 = MessageCell.createUserMessageCell(List.of(text2));
        assertNotEquals(cell1, cell2);
    }

    @Test
    void testPngContentPart() {
        byte[] data = {1,2,3};
        PngContentPart png = new PngContentPart(data);
        MessageCell cell = MessageCell.createAIMessageCell(List.of(png));
        assertEquals(MessageType.AI, cell.getMessageType());
        assertEquals(png, cell.getContent().get(0));
        assertArrayEquals(data, ((PngContentPart)cell.getContent().get(0)).getData());
    }

    @Test
    void testNullContentThrows() {
        Exception ex = assertThrows(NullPointerException.class, () -> {
            MessageCell.createUserMessageCell(null);
        });
        assertTrue(ex.getMessage().contains("Content cannot be null"));
    }

    @Test
    void testEmptyContentThrows() {
        Exception ex = assertThrows(IllegalArgumentException.class, () -> {
            MessageCell.createUserMessageCell(List.of());
        });
        assertTrue(ex.getMessage().contains("Content cannot be empty"));
    }

    @Test
    void testNullToolCallIdThrows() {
        TextContentPart text = new TextContentPart("Tool");
        Exception ex = assertThrows(NullPointerException.class, () -> {
            MessageCell.createToolMessageCell(List.of(text), null, "toolname");
        });
        assertTrue(ex.getMessage().contains("Tool call ID cannot be null"));
    }
}
