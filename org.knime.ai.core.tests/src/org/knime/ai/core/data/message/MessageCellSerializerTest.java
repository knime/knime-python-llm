package org.knime.ai.core.data.message;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.knime.ai.core.data.message.MessageValue.ToolCall;
import org.knime.core.data.DataCell;
import org.knime.core.data.DataCellDataInput;
import org.knime.core.data.DataCellDataOutput;

class MessageCellSerializerTest {

    private MessageCellSerializer serializer = new MessageCellSerializer();

    private static class MockDataCellDataOutput extends DataOutputStream implements DataCellDataOutput {
        MockDataCellDataOutput(final ByteArrayOutputStream out) { super(out); }
        @Override public void writeDataCell(final DataCell cell) { throw new UnsupportedOperationException(); }
    }
    private static class MockDataCellDataInput extends DataInputStream implements DataCellDataInput {
        MockDataCellDataInput(final ByteArrayInputStream in) { super(in); }
        @Override public DataCell readDataCell() { throw new UnsupportedOperationException(); }
    }

    @Test
    void testSerializeDeserialize_textContent() throws Exception {
        MessageCell cell = MessageCell.createUserMessageCell(List.of(new TextContentPart("Hello World!")));
        MessageCell roundtrip = serializeAndDeserialize(cell);
        assertEquals(cell.getMessageType(), roundtrip.getMessageType());
        assertEquals(cell.getContent().size(), roundtrip.getContent().size());
        assertEquals(((TextContentPart)cell.getContent().get(0)).getContent(), ((TextContentPart)roundtrip.getContent().get(0)).getContent());
        assertTrue(roundtrip.getToolCalls().isEmpty());
        assertTrue(roundtrip.getToolCallId().isEmpty());
        assertTrue(roundtrip.getName().isEmpty());
    }

    @Test
    void testSerializeDeserialize_pngContent() throws Exception {
        byte[] image = new byte[] {1,2,3,4,5};
        MessageCell cell = MessageCell.createAIMessageCell(List.of(new PngContentPart(image)));
        MessageCell roundtrip = serializeAndDeserialize(cell);
        assertEquals(cell.getMessageType(), roundtrip.getMessageType());
        assertArrayEquals(((PngContentPart)cell.getContent().get(0)).getData(), ((PngContentPart)roundtrip.getContent().get(0)).getData());
    }

    @Test
    void testSerializeDeserialize_withToolCalls() throws Exception {
        ToolCall toolCall = new ToolCall("myTool", "id123", "{\"param\":42}");
        MessageCell cell = MessageCell.createAIMessageCell(List.of(new TextContentPart("Tool call test")), List.of(toolCall));
        MessageCell roundtrip = serializeAndDeserialize(cell);
        assertEquals(1, roundtrip.getToolCalls().get().size());
        ToolCall roundtripCall = roundtrip.getToolCalls().get().get(0);
        assertEquals(toolCall.toolName(), roundtripCall.toolName());
        assertEquals(toolCall.id(), roundtripCall.id());
        assertEquals(toolCall.arguments(), roundtripCall.arguments());
    }

    @Test
    void testSerializeDeserialize_toolMessageCell() throws Exception {
        MessageCell cell = MessageCell.createToolMessageCell(List.of(new TextContentPart("Tool response")), "callid", "toolname");
        MessageCell roundtrip = serializeAndDeserialize(cell);
        assertEquals("callid", roundtrip.getToolCallId().get());
        assertEquals("toolname", roundtrip.getName().get());
    }

    private MessageCell serializeAndDeserialize(final MessageCell cell) throws Exception {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        MockDataCellDataOutput out = new MockDataCellDataOutput(baos);
        serializer.serialize(cell, out);
        out.flush();
        ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
        MockDataCellDataInput in = new MockDataCellDataInput(bais);
        return serializer.deserialize(in);
    }
}
