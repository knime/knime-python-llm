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
 *   May 28, 2025 (Adrian Nembach, KNIME GmbH, Konstanz, Germany): created
 */
package org.knime.ai.core.data.message;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;

import org.junit.jupiter.api.Test;
import org.knime.core.table.access.ListAccess.ListReadAccess;
import org.knime.core.table.access.ListAccess.ListWriteAccess;
import org.knime.core.table.access.ReadAccess;
import org.knime.core.table.access.WriteAccess;

@SuppressWarnings("static-method")
final class ValueFactoryUtilsTest {

    @Test
    void testChain() {
        Supplier<String> source = () -> "42";
        Function<String, Integer> mapper = Integer::parseInt;
        Supplier<Integer> chained = ValueFactoryUtils.chain(source, mapper);
        assertEquals(42, chained.get());
    }

    @Test
    void testReadList() {
        ListReadAccess listAccess = mock(ListReadAccess.class);
        when(listAccess.size()).thenReturn(3);
        @SuppressWarnings("unchecked")
        Supplier<String> itemReader = mock(Supplier.class);
        when(itemReader.get()).thenReturn("a", "b", "c");
        // Simulate setIndex does nothing
        doNothing().when(listAccess).setIndex(anyInt());
        Supplier<List<String>> reader = ValueFactoryUtils.readList(listAccess, itemReader);
        List<String> result = reader.get();
        assertEquals(Arrays.asList("a", "b", "c"), result);
        verify(listAccess, times(3)).setIndex(anyInt());
        verify(itemReader, times(3)).get();
    }

    @Test
    void testReadOptionalPresent() {
        ReadAccess access = mock(ReadAccess.class);
        when(access.isMissing()).thenReturn(false);
        Supplier<String> valueReader = () -> "foo";
        Supplier<Optional<String>> reader = ValueFactoryUtils.readOptional(access, valueReader);
        Optional<String> result = reader.get();
        assertTrue(result.isPresent());
        assertEquals("foo", result.get());
    }

    @Test
    void testReadOptionalMissing() {
        ReadAccess access = mock(ReadAccess.class);
        when(access.isMissing()).thenReturn(true);
        Supplier<String> valueReader = () -> "foo";
        Supplier<Optional<String>> reader = ValueFactoryUtils.readOptional(access, valueReader);
        Optional<String> result = reader.get();
        assertFalse(result.isPresent());
    }

    @Test
    void testWriteList() {
        ListWriteAccess access = mock(ListWriteAccess.class);
        @SuppressWarnings("unchecked")
        Consumer<String> itemWriter = mock(Consumer.class);
        List<String> list = Arrays.asList("x", null, "z");
        Consumer<List<String>> writer = ValueFactoryUtils.writeList(access, itemWriter);
        writer.accept(list);
        verify(access).create(3);
        verify(access, times(3)).setWriteIndex(anyInt());
        verify(itemWriter).accept("x");
        verify(itemWriter).accept("z");
        verify(access, times(1)).setMissing();
    }

    @Test
    void testWriteOptionalPresent() {
        WriteAccess access = mock(WriteAccess.class);
        @SuppressWarnings("unchecked")
        Consumer<String> valueWriter = mock(Consumer.class);
        Consumer<Optional<String>> writer = ValueFactoryUtils.writeOptional(access, valueWriter);
        writer.accept(Optional.of("bar"));
        verify(valueWriter).accept("bar");
        verify(access, never()).setMissing();
    }

    @Test
    void testWriteOptionalEmpty() {
        WriteAccess access = mock(WriteAccess.class);
        @SuppressWarnings("unchecked")
        Consumer<String> valueWriter = mock(Consumer.class);
        Consumer<Optional<String>> writer = ValueFactoryUtils.writeOptional(access, valueWriter);
        writer.accept(Optional.empty());
        verify(valueWriter, never()).accept(any());
        verify(access).setMissing();
    }
}
