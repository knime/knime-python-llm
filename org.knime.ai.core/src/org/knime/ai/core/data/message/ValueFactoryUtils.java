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

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;

import org.knime.core.table.access.ListAccess.ListReadAccess;
import org.knime.core.table.access.ListAccess.ListWriteAccess;
import org.knime.core.table.access.ReadAccess;
import org.knime.core.table.access.WriteAccess;

/**
 * Contains utility methods for implemtning a ValueFactory.
 *
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 */
final class ValueFactoryUtils {

    private ValueFactoryUtils() {
        // Utility class, no instantiation
    }

    static <S, T> Supplier<T> chain(final Supplier<S> source, final Function<S, T> mapper) {
        return () -> mapper.apply(source.get());
    }

    static <T> Supplier<List<T>> readList(final ListReadAccess listAccess, final Supplier<T> itemReader) {
        return () -> {
            var list = new ArrayList<T>();
            var numItems = listAccess.size();
            for (int i = 0; i < numItems; i++) {
                listAccess.setIndex(i);
                list.add(itemReader.get());
            }
            return list;
        };
    }

    static <T> Supplier<Optional<T>> readOptional(final ReadAccess access,
            final Supplier<T> valueReader) {
        return () -> access.isMissing() ? Optional.empty() : Optional.of(valueReader.get());
    }

    static <T> Consumer<List<T>> writeList(final ListWriteAccess access, final Consumer<T> itemWriter) {
        return (list) -> {
            for (int i = 0; i < list.size(); i++) {
                access.setWriteIndex(i);
                var item = list.get(i);
                if (item == null) {
                    access.setMissing();
                } else {
                    itemWriter.accept(item);
                }
            }
        };
    }

    static <T> Consumer<Optional<T>> writeOptional(final WriteAccess access, final Consumer<T> valueWriter) {
        return (value) -> value.ifPresentOrElse(valueWriter, access::setMissing);
    }

}
