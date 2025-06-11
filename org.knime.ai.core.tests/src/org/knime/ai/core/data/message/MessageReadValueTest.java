package org.knime.ai.core.data.message;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.Optional;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.knime.ai.core.data.message.MessageValue.MessageType;
import org.knime.core.data.DataCell;
import org.knime.core.table.access.ListAccess.ListReadAccess;
import org.knime.core.table.access.StringAccess.StringReadAccess;
import org.knime.core.table.access.StructAccess.StructReadAccess;
import org.knime.core.table.access.VarBinaryAccess.VarBinaryReadAccess;

final class MessageReadValueTest {

    private StructReadAccess structAccess;
    private StringReadAccess typeAccess;
    private ListReadAccess contentAccess;
    private ListReadAccess toolCallsAccess;
    private StringReadAccess toolCallIdAccess;
    private StringReadAccess nameAccess;
    private StructReadAccess contentStructAccess;
    private StringReadAccess contentPartTypeAccess;
    private VarBinaryReadAccess contentPartDataAccess;
    private StructReadAccess toolCallsStructAccess;
    private StringReadAccess toolCallsIdAccess;
    private StringReadAccess toolCallsNameAccess;
    private StringReadAccess toolCallsArgsAccess;


    @BeforeEach
    void setUp() {
        structAccess = mock(StructReadAccess.class);
        typeAccess = mock(StringReadAccess.class);
        contentAccess = mock(ListReadAccess.class);
        toolCallsAccess = mock(ListReadAccess.class);
        toolCallIdAccess = mock(StringReadAccess.class);
        nameAccess = mock(StringReadAccess.class);
        contentPartTypeAccess = mock(StringReadAccess.class);
        contentPartDataAccess = mock(VarBinaryReadAccess.class);
        toolCallsIdAccess = mock(StringReadAccess.class);
        toolCallsNameAccess = mock(StringReadAccess.class);
        toolCallsArgsAccess = mock(StringReadAccess.class);

        when(structAccess.size()).thenReturn(5);
        when(structAccess.getAccess(0)).thenReturn(typeAccess);
        when(structAccess.getAccess(1)).thenReturn(contentAccess);
        when(structAccess.getAccess(2)).thenReturn(toolCallsAccess);
        when(structAccess.getAccess(3)).thenReturn(toolCallIdAccess);
        when(structAccess.getAccess(4)).thenReturn(nameAccess);

        when(structAccess.getAccess(0)).thenReturn(typeAccess);
        when(structAccess.getAccess(1)).thenReturn(contentAccess);
        when(structAccess.getAccess(2)).thenReturn(toolCallsAccess);
        when(structAccess.getAccess(3)).thenReturn(toolCallIdAccess);
        when(structAccess.getAccess(4)).thenReturn(nameAccess);

        contentStructAccess = mock(StructReadAccess.class);
        when(contentAccess.getAccess()).thenReturn(contentStructAccess);
        when(contentStructAccess.getAccess(0)).thenReturn(contentPartTypeAccess);
        when(contentStructAccess.getAccess(1)).thenReturn(contentPartDataAccess);

        toolCallsStructAccess = mock(StructReadAccess.class);
        when(toolCallsAccess.getAccess()).thenReturn(toolCallsStructAccess);
        when(toolCallsStructAccess.getAccess(0)).thenReturn(toolCallsIdAccess);
        when(toolCallsStructAccess.getAccess(1)).thenReturn(toolCallsNameAccess);
        when(toolCallsStructAccess.getAccess(2)).thenReturn(toolCallsArgsAccess);
    }

    @Test
    void testGetMessageType() {
        when(typeAccess.getStringValue()).thenReturn("USER");
        MessageReadValue value = new MessageReadValue(structAccess);
        assertEquals(MessageType.USER, value.getMessageType());
    }

    @Test
    void testGetToolCallId() {
        when(toolCallIdAccess.getStringValue()).thenReturn("id123");
        MessageReadValue value = new MessageReadValue(structAccess);
        assertEquals(Optional.of("id123"), value.getToolCallId());
    }

    @Test
    void testGetName() {
        when(nameAccess.getStringValue()).thenReturn("toolName");
        MessageReadValue value = new MessageReadValue(structAccess);
        assertEquals(Optional.of("toolName"), value.getName());
    }

    @Test
    void testGetContent_Empty() {
        when(contentAccess.getAccess()).thenReturn(mock(StructReadAccess.class));
        when(contentAccess.size()).thenReturn(0);
        MessageReadValue value = new MessageReadValue(structAccess);
        assertNotNull(value.getContent());
        assertTrue(value.getContent().isEmpty());
    }

    @Test
    void testGetToolCalls_Empty() {
        when(toolCallsAccess.getAccess()).thenReturn(mock(StructReadAccess.class));
        when(toolCallsAccess.size()).thenReturn(0);
        MessageReadValue value = new MessageReadValue(structAccess);
        assertTrue(value.getToolCalls().isEmpty() || value.getToolCalls().get().isEmpty());
    }

    @Test
    void testGetDataCell_NotNull() {
        when(typeAccess.getStringValue()).thenReturn("USER");
        when(contentAccess.getAccess()).thenReturn(mock(StructReadAccess.class));
        when(contentAccess.size()).thenReturn(0);
        when(toolCallIdAccess.isMissing()).thenReturn(true);
        when(nameAccess.isMissing()).thenReturn(true);
        MessageReadValue value = new MessageReadValue(structAccess);
        DataCell cell = value.getDataCell();
        assertNotNull(cell);
    }

    @Test
    void testGetContent_TextContentPart() {
        String text = "Hello World!";
        when(contentAccess.size()).thenReturn(1);
        when(contentPartTypeAccess.getStringValue()).thenReturn("text");
        when(contentPartDataAccess.getByteArray()).thenReturn(text.getBytes());

        MessageReadValue value = new MessageReadValue(structAccess);
        var content = value.getContent();
        assertEquals(1, content.size());
        assertInstanceOf(TextContentPart.class, content.get(0));
        assertEquals(text, ((TextContentPart)content.get(0)).getContent());
    }

    @Test
    void testGetContent_PngContentPart() {
        byte[] pngData = new byte[] {1, 2, 3, 4};
        when(contentAccess.size()).thenReturn(1);
        when(contentPartTypeAccess.getStringValue()).thenReturn("png");
        when(contentPartDataAccess.getByteArray()).thenReturn(pngData);

        MessageReadValue value = new MessageReadValue(structAccess);
        var content = value.getContent();
        assertEquals(1, content.size());
        assertInstanceOf(PngContentPart.class, content.get(0));
        assertArrayEquals(pngData, ((PngContentPart)content.get(0)).getData());
    }

}
