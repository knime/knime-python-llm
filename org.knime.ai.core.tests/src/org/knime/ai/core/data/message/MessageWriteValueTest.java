package org.knime.ai.core.data.message;

import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.Optional;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.knime.ai.core.data.message.MessageValue.MessageContentPart;
import org.knime.ai.core.data.message.MessageValue.MessageType;
import org.knime.ai.core.data.message.MessageValue.ToolCall;
import org.knime.core.table.access.ListAccess.ListWriteAccess;
import org.knime.core.table.access.StringAccess.StringWriteAccess;
import org.knime.core.table.access.StructAccess.StructWriteAccess;
import org.knime.core.table.access.VarBinaryAccess.VarBinaryWriteAccess;

class MessageWriteValueTest {

    private StructWriteAccess structAccess;
    private StringWriteAccess typeAccess;
    private ListWriteAccess contentAccess;
    private ListWriteAccess toolCallsAccess;
    private StringWriteAccess toolCallIdAccess;
    private StringWriteAccess nameAccess;

    @BeforeEach
    void setup() {
        structAccess = mock(StructWriteAccess.class);
        typeAccess = mock(StringWriteAccess.class);
        contentAccess = mock(ListWriteAccess.class);
        toolCallsAccess = mock(ListWriteAccess.class);
        toolCallIdAccess = mock(StringWriteAccess.class);
        nameAccess = mock(StringWriteAccess.class);

        when(structAccess.getWriteAccess(0)).thenReturn(typeAccess);
        when(structAccess.getWriteAccess(1)).thenReturn(contentAccess);
        when(structAccess.getWriteAccess(2)).thenReturn(toolCallsAccess);
        when(structAccess.getWriteAccess(3)).thenReturn(toolCallIdAccess);
        when(structAccess.getWriteAccess(4)).thenReturn(nameAccess);
    }

    @Test
    void testSetValue_writesAllFields() {
        // Arrange
        MessageContentPart contentPart = mock(MessageContentPart.class);
        when(contentPart.getType()).thenReturn(MessageValue.MessageContentPart.MessageContentPartType.TEXT);
        when(contentPart.getData()).thenReturn("hello".getBytes());
        List<MessageContentPart> contentList = List.of(contentPart);

        ToolCall toolCall = new ToolCall("tool", "id", "args");
        List<ToolCall> toolCalls = List.of(toolCall);
        Optional<List<ToolCall>> toolCallsOpt = Optional.of(toolCalls);
        Optional<String> toolCallIdOpt = Optional.of("toolCallId");
        Optional<String> nameOpt = Optional.of("name");

        MessageValue value = mock(MessageValue.class);
        when(value.getMessageType()).thenReturn(MessageType.USER);
        when(value.getContent()).thenReturn(contentList);
        when(value.getToolCalls()).thenReturn(toolCallsOpt);
        when(value.getToolCallId()).thenReturn(toolCallIdOpt);
        when(value.getName()).thenReturn(nameOpt);

        // Mock content part struct access
        StructWriteAccess contentPartAccess = mock(StructWriteAccess.class);
        StringWriteAccess contentTypeAccess = mock(StringWriteAccess.class);
        VarBinaryWriteAccess contentDataAccess = mock(VarBinaryWriteAccess.class);
        when(contentAccess.getWriteAccess()).thenReturn(contentPartAccess);
        when(contentPartAccess.getWriteAccess(0)).thenReturn(contentTypeAccess);
        when(contentPartAccess.getWriteAccess(1)).thenReturn(contentDataAccess);

        // Mock tool call struct access
        StructWriteAccess toolCallStructAccess = mock(StructWriteAccess.class);
        StringWriteAccess toolCallIdField = mock(StringWriteAccess.class);
        StringWriteAccess toolCallNameField = mock(StringWriteAccess.class);
        StringWriteAccess toolCallArgsField = mock(StringWriteAccess.class);
        when(toolCallsAccess.getWriteAccess()).thenReturn(toolCallStructAccess);
        when(toolCallStructAccess.getWriteAccess(0)).thenReturn(toolCallIdField);
        when(toolCallStructAccess.getWriteAccess(1)).thenReturn(toolCallNameField);
        when(toolCallStructAccess.getWriteAccess(2)).thenReturn(toolCallArgsField);

        MessageWriteValue writeValue = new MessageWriteValue(structAccess);

        // Act
        writeValue.setValue(value);

        // Assert
        verify(typeAccess).setStringValue("USER");
        verify(contentAccess).create(1);
        verify(contentAccess).setWriteIndex(0);
        verify(contentTypeAccess).setStringValue("text");
        verify(contentDataAccess).setByteArray("hello".getBytes());
        verify(toolCallsAccess).create(1);
        verify(toolCallsAccess).setWriteIndex(0);
        verify(toolCallIdField).setStringValue("id");
        verify(toolCallNameField).setStringValue("tool");
        verify(toolCallArgsField).setStringValue("args");
        verify(toolCallIdAccess).setStringValue("toolCallId");
        verify(nameAccess).setStringValue("name");
    }

    @Test
    void testSetValue_handlesEmptyOptionalsAndNullLists() {
        MessageValue value = mock(MessageValue.class);
        when(value.getMessageType()).thenReturn(MessageType.AI);
        when(value.getContent()).thenReturn(List.of());
        when(value.getToolCalls()).thenReturn(Optional.empty());
        when(value.getToolCallId()).thenReturn(Optional.empty());
        when(value.getName()).thenReturn(Optional.empty());

        // Mock content part struct access
        StructWriteAccess contentPartAccess = mock(StructWriteAccess.class);
        when(contentAccess.getWriteAccess()).thenReturn(contentPartAccess);
        // Mock tool call struct access (needed even if not used)
        StructWriteAccess toolCallStructAccess = mock(StructWriteAccess.class);
        when(toolCallsAccess.getWriteAccess()).thenReturn(toolCallStructAccess);
        // No content parts, so no further setup needed

        MessageWriteValue writeValue = new MessageWriteValue(structAccess);
        writeValue.setValue(value);

        verify(typeAccess).setStringValue("AI");
        verify(contentAccess).create(0);
        verify(toolCallsAccess).setMissing();
        verify(toolCallIdAccess).setMissing();
        verify(nameAccess).setMissing();
    }
}
