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
 *   Jun 6, 2025 (Adrian Nembach, KNIME GmbH, Konstanz, Germany): created
 */
package org.knime.ai.core.node.message;

import java.util.List;
import java.util.function.BinaryOperator;

import org.knime.core.data.StringValue;
import org.knime.core.data.image.png.PNGImageValue;
import org.knime.core.data.json.JSONValue;
import org.knime.core.webui.node.dialog.defaultdialog.widget.choices.column.CompatibleColumnsProvider;
import org.knime.core.webui.node.dialog.defaultdialog.widget.updates.Predicate;
import org.knime.core.webui.node.dialog.defaultdialog.widget.updates.PredicateProvider;

/**
 * Contains utility classes for settings.
 *
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 */
public final class SettingsUtils {

    /**
     * A composite predicate provider that combines multiple predicate providers into a single predicate.
     *
     * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
     */
    public abstract static class CompositePredicateProvider implements PredicateProvider {

        private final List<PredicateProvider> m_predicates;

        private final BinaryOperator<Predicate> m_aggregator;

        /**
         * Creates a new composite predicate provider that combines the given predicates using the given accumulator.
         *
         * @param accumulator used to accumulate the predicates (starting with the initializer's always predicate)
         * @param predicates the predicates to combine
         */
        protected CompositePredicateProvider(final BinaryOperator<Predicate> accumulator,
            final PredicateProvider... predicates) {
            if (predicates == null || predicates.length == 0) {
                throw new IllegalArgumentException("At least one predicate must be provided.");
            }
            m_predicates = List.of(predicates);
            m_aggregator = accumulator;
        }

        /**
         * Default constructor that combines the given predicates using a logical AND operation.
         *
         * @param predicates the predicates to combine
         */
        protected CompositePredicateProvider(final PredicateProvider... predicates) {
            this(Predicate::and, predicates);
        }

        @Override
        public Predicate init(final PredicateInitializer i) {
            return m_predicates.stream()
                    .map(p -> p.init(i))
                    .reduce(m_aggregator)
                    .orElseThrow(() -> new IllegalStateException("No predicates were provided."));
        }

    }

    /**
     * Utility class that provides compatible column providers for different data types.
     *
     * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
     */
    public static final class ColumnProviders {

        /**
         * Provides compatible columns for PNG image values.
         *
         * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
         */
        public static final class PngColumns extends CompatibleColumnsProvider {

            /**
             * Constructor for PngColumns.
             */
            protected PngColumns() {
                super(PNGImageValue.class);
            }

        }

        /**
         * Provides compatible columns for string values.
         *
         * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
         */
        public static final class StringColumns extends CompatibleColumnsProvider {

            /**
             * Constructor for StringColumns.
             */
            protected StringColumns() {
                super(StringValue.class);
            }

        }

        /**
         * Provides compatible columns for JSON values.
         *
         * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
         */
        public static final class JsonColumns extends CompatibleColumnsProvider {

            /**
             * Constructor for JsonColumns.
             */
            protected JsonColumns() {
                super(JSONValue.class);
            }

        }
    }
}
