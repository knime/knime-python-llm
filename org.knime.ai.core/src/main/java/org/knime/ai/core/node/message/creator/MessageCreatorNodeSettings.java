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
package org.knime.ai.core.node.message.creator;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.IntStream;

import org.knime.ai.core.data.message.MessageValue;
import org.knime.ai.core.data.message.MessageValue.MessageType;
import org.knime.core.data.DataColumnSpec;
import org.knime.core.data.image.png.PNGImageValue;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.workflow.NodeContainer;
import org.knime.core.node.workflow.NodeContext;
import org.knime.core.webui.node.dialog.configmapping.ConfigMigration;
import org.knime.core.webui.node.dialog.defaultdialog.DefaultNodeSettings;
import org.knime.core.webui.node.dialog.defaultdialog.layout.After;
import org.knime.core.webui.node.dialog.defaultdialog.layout.HorizontalLayout;
import org.knime.core.webui.node.dialog.defaultdialog.layout.Layout;
import org.knime.core.webui.node.dialog.defaultdialog.persistence.api.Migration;
import org.knime.core.webui.node.dialog.defaultdialog.persistence.api.NodeSettingsMigration;
import org.knime.core.webui.node.dialog.defaultdialog.widget.ArrayWidget;
import org.knime.core.webui.node.dialog.defaultdialog.widget.Label;
import org.knime.core.webui.node.dialog.defaultdialog.widget.TextAreaWidget;
import org.knime.core.webui.node.dialog.defaultdialog.widget.ValueSwitchWidget;
import org.knime.core.webui.node.dialog.defaultdialog.widget.Widget;
import org.knime.core.webui.node.dialog.defaultdialog.widget.choices.ChoicesProvider;
import org.knime.core.webui.node.dialog.defaultdialog.widget.choices.StringChoice;
import org.knime.core.webui.node.dialog.defaultdialog.widget.choices.StringChoicesProvider;
import org.knime.core.webui.node.dialog.defaultdialog.widget.choices.column.FilteredInputTableColumnsProvider;
import org.knime.core.webui.node.dialog.defaultdialog.widget.updates.Effect;
import org.knime.core.webui.node.dialog.defaultdialog.widget.updates.Effect.EffectType;
import org.knime.core.webui.node.dialog.defaultdialog.widget.updates.Predicate;
import org.knime.core.webui.node.dialog.defaultdialog.widget.updates.PredicateProvider;
import org.knime.core.webui.node.dialog.defaultdialog.widget.updates.Reference;
import org.knime.core.webui.node.dialog.defaultdialog.widget.updates.StateProvider;
import org.knime.core.webui.node.dialog.defaultdialog.widget.updates.ValueReference;

/**
 *
 * @author Seray Arslan, KNIME GmbH, Konstanz, Germany
 */
@SuppressWarnings("restriction")
final class MessageCreatorNodeSettings implements DefaultNodeSettings {
    @Widget(title = "Role",
            description = "Message role")
    @ValueReference(RoleRef.class)
    MessageType m_role;

    static final class RoleRef implements Reference<MessageType> {
    }

    @Widget(title = "Message Content", description = "Define the content parts (text, image) of the message.")
    @ArrayWidget(
        addButtonText = "Add content",
        showSortButtons = false,
        elementTitle = "Content"
    )
    @ValueReference(ContentRef.class)
    @Migration(ContentMigrator.class)
    Contents[] m_content = new Contents[]{
        new Contents()
    };

    static final class ContentRef implements Reference<Contents[]> {
    }

    static final class Contents implements DefaultNodeSettings {

        enum TextOrImage {
            @Label("Text")
            TEXT,

            @Label("Image")
            IMAGE;
        }

        @HorizontalLayout
        interface ContentsLayout {
        }

        @Layout(ContentsLayout.class)
        @Widget(title = "Content Type", description = "Select the type of content (Text or Image).")
        @ValueSwitchWidget
        @ValueReference(ContentTypeRef.class)
        TextOrImage m_textOrImage = TextOrImage.TEXT;

        static final class ContentTypeRef implements Reference<TextOrImage> {

        }

        @After(ContentsLayout.class)
        interface ContentsValueLayout {
        }

        static final class IsTextTypePredicate implements PredicateProvider {

            @Override
            public Predicate init(final PredicateInitializer i) {
                return i.getEnum(ContentTypeRef.class).isOneOf(TextOrImage.TEXT);
            }

        }

        @Layout(ContentsValueLayout.class)
        @Widget(title = "Image Column", description = "Select the input table column containing PNG images.")
        @ChoicesProvider(ImageCellColumns.class)
        @Effect(predicate = IsTextTypePredicate.class, type = EffectType.HIDE)
        String m_imageColumn;


        @Layout(ContentsValueLayout.class)
        @Widget(title = "Text Value", description = "Enter the text content.")
        @TextAreaWidget
        @Effect(predicate = IsTextTypePredicate.class, type = EffectType.SHOW)
        String m_textValue;

        Contents(final String textValue, final String imageColumn) {
            m_textValue = textValue;
            m_imageColumn = imageColumn;
        }

        Contents() {
            this("", "");
        }

        static final class ImageCellColumns implements FilteredInputTableColumnsProvider {

            @Override
            public boolean isIncluded(final DataColumnSpec col) {
                return isPNGCell(col);
            }
            static boolean isPNGCell(final DataColumnSpec colSpec) {
                final var type = colSpec.getType();
                return type.isCompatible(PNGImageValue.class);
            }

        }
        static final class DefaultContentSettingsProvider implements StateProvider<Contents> {

            private Supplier<Contents[]> m_valueSupplier;

            @Override
            public void init(final StateProviderInitializer initializer) {
                initializer.computeAfterOpenDialog();
                m_valueSupplier = initializer.computeFromValueSupplier(ContentRef.class);
            }

            @Override
            public Contents computeState(final DefaultNodeSettingsContext context) {
                return new Contents("", "");
            }
        }

    }


    static final class ContentMigrator implements NodeSettingsMigration<Contents[]> {

        private static final String CFG_KEY_VARIABLES = "variables";
        private static final String CFG_SUBKEY_TYPES = "types";
        private static final String CFG_SUBKEY_VALUES = "values";

        @Override
        public List<ConfigMigration<Contents[]>> getConfigMigrations() {
            var builder = ConfigMigration.builder(ContentMigrator::load);
            return List.of(builder.build());
        }

        static Contents[] load(final NodeSettingsRO settings) throws InvalidSettingsException {
            var subConfig = settings.getConfig(CFG_KEY_VARIABLES);
            var types = subConfig.getStringArray(CFG_SUBKEY_TYPES);
            var values = subConfig.getStringArray(CFG_SUBKEY_VALUES);

            if (types.length != values.length) {
                throw new InvalidSettingsException("types and values must have the same length");
            }

            return IntStream.range(0, types.length)
                .mapToObj(i -> new Contents(
                    types[i],
                    values[i]
                ))
                .toArray(Contents[]::new);
        }
    }

    private static class RoleChoices implements StringChoicesProvider {

        @Override
        public void init(final StateProviderInitializer initializer) {
            initializer.computeBeforeOpenDialog();
        }

        @Override
        public List<StringChoice> computeState(final DefaultNodeSettingsContext context) {
            var nc = NodeContext.getContext().getNodeContainer();
            var wfm = nc.getParent();

            return wfm.getNodeContainers().stream()
                    .filter(n -> nc != n)//
                    .map(RoleChoices::createChoice)
                    .toList();
        }


        private static StringChoice createChoice(final NodeContainer nc) {
            MessageType[] roles = MessageValue.MessageType.values();
            var rolesAsString = Arrays.stream(roles).map(Enum::name).toArray(String[]::new);
            var text = nc.getName() + " (" + nc.getID() + ")";
            return new StringChoice(String.join(", ", rolesAsString), text);
        }
    }

    static final class Content implements DefaultNodeSettings {
        @Widget(title = "Role", description = "The role for the message.")
        @ChoicesProvider(RoleChoices.class)
        String m_role;

        @Widget(title = "Content", description = "The content for the message.")
        String m_content;
    }
}
