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

import org.knime.ai.core.data.message.MessageValue.MessageType;
import org.knime.ai.core.node.message.creator.SettingsUtils.CompositePredicateProvider;
import org.knime.core.data.StringValue;
import org.knime.core.data.image.png.PNGImageValue;
import org.knime.core.webui.node.dialog.defaultdialog.DefaultNodeSettings;
import org.knime.core.webui.node.dialog.defaultdialog.layout.After;
import org.knime.core.webui.node.dialog.defaultdialog.layout.HorizontalLayout;
import org.knime.core.webui.node.dialog.defaultdialog.layout.Layout;
import org.knime.core.webui.node.dialog.defaultdialog.setting.singleselection.NoneChoice;
import org.knime.core.webui.node.dialog.defaultdialog.setting.singleselection.StringOrEnum;
import org.knime.core.webui.node.dialog.defaultdialog.widget.ArrayWidget;
import org.knime.core.webui.node.dialog.defaultdialog.widget.Label;
import org.knime.core.webui.node.dialog.defaultdialog.widget.TextAreaWidget;
import org.knime.core.webui.node.dialog.defaultdialog.widget.ValueSwitchWidget;
import org.knime.core.webui.node.dialog.defaultdialog.widget.Widget;
import org.knime.core.webui.node.dialog.defaultdialog.widget.choices.ChoicesProvider;
import org.knime.core.webui.node.dialog.defaultdialog.widget.choices.column.CompatibleColumnsProvider;
import org.knime.core.webui.node.dialog.defaultdialog.widget.updates.Effect;
import org.knime.core.webui.node.dialog.defaultdialog.widget.updates.Effect.EffectType;
import org.knime.core.webui.node.dialog.defaultdialog.widget.updates.Predicate;
import org.knime.core.webui.node.dialog.defaultdialog.widget.updates.Reference;
import org.knime.core.webui.node.dialog.defaultdialog.widget.updates.ValueReference;

/**
 * Settings for the Message Creator node.
 *
 * @author Seray Arslan, KNIME GmbH, Konstanz, Germany
 */
@SuppressWarnings("restriction")
final class MessageCreatorNodeSettings implements DefaultNodeSettings {
    enum InputType {
        @Label("Value")
        VALUE,
        @Label("Column")
        COLUMN;
    }

    static final class RoleInputTypeRef implements Reference<InputType> {}

    private static final class IsValueRoleInputType extends CompositePredicateProvider {

        IsValueRoleInputType() {
            super(i -> i.getEnum(RoleInputTypeRef.class).isOneOf(InputType.VALUE));
        }

    }

    @Widget(title = "Input Type",
            description = "Select how the message role is provided (as a value or as a column).")
    @ValueSwitchWidget
    @ValueReference(RoleInputTypeRef.class)
    InputType m_roleInputType = InputType.VALUE;

    @Widget(title = "Role",
            description = "Message role")
    @Effect(predicate = IsValueRoleInputType.class, type = EffectType.SHOW)
    MessageType m_roleValue = MessageType.USER;

    @Widget(title = "Role Column",
            description = "Select the input table column containing message roles.")
    @Effect(predicate = IsValueRoleInputType.class, type = EffectType.HIDE)
    @ChoicesProvider(TextCellColumns.class)
    String m_roleColumn;

    @Widget(title = "Tool Call Id Column",
            description = "(Optional) Select the input table column containing the tool call id.")
    @ChoicesProvider(TextCellColumns.class)
    StringOrEnum<NoneChoice> m_toolCallIdColumn = new StringOrEnum<>(NoneChoice.NONE);

    @Widget(title = "Message Content", description = "Define the content parts (text, image) of the message.")
    @ArrayWidget(
        addButtonText = "Add content",
        showSortButtons = false,
        elementTitle = "Content"
    )
    Contents[] m_content = new Contents[]{
        new Contents()
    };


    static final class Contents implements DefaultNodeSettings {

        enum ContentType {
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
        ContentType m_contentType = ContentType.TEXT;

        @Layout(ContentsLayout.class)
        @Widget(title = "Input Type", description = "Select how the content is provided (as a value or as a column).")
        @ValueSwitchWidget
        @ValueReference(InputTypeRef.class)
        @Effect(predicate = HasInputTypes.class, type = EffectType.SHOW)
        InputType m_inputType = InputType.VALUE;

        @Layout(ContentsValueLayout.class)
        @Widget(title = "Image Column", description = "Select the input table column containing PNG images.")
        @ChoicesProvider(ImageCellColumns.class)
        @Effect(predicate = IsImageValue.class, type = EffectType.SHOW)
        String m_imageColumn;


        @Layout(ContentsValueLayout.class)
        @Widget(title = "Text Value", description = "Enter the text content.")
        @TextAreaWidget
        @Effect(predicate = IsTextValue.class, type = EffectType.SHOW)
        String m_textValue;

        @Layout(ContentsValueLayout.class)
        @Widget(title = "Text Column", description = "Select the input table column containing text values.")
        @Effect(predicate = IsTextColumn.class, type = EffectType.SHOW)
        @ChoicesProvider(TextCellColumns.class)
        String m_textColumn;


        static final class ContentTypeRef implements Reference<ContentType> {

        }

        static final class InputTypeRef implements Reference<InputType> {

        }

        @After(ContentsLayout.class)
        interface ContentsValueLayout {
        }

        static final class HasInputTypes extends CompositePredicateProvider {

            HasInputTypes() {
                super(i -> i.getEnum(ContentTypeRef.class).isOneOf(ContentType.TEXT));
            }

        }

        static final class IsTextValue extends CompositePredicateProvider {

            IsTextValue() {
                super(Predicate::and,
                    i -> i.getEnum(ContentTypeRef.class).isOneOf(ContentType.TEXT),
                    i -> i.getEnum(InputTypeRef.class).isOneOf(InputType.VALUE));
            }

        }

        static final class IsTextColumn extends CompositePredicateProvider {

            IsTextColumn() {
                super(Predicate::and,
                    i -> i.getEnum(ContentTypeRef.class).isOneOf(ContentType.TEXT),
                    i -> i.getEnum(InputTypeRef.class).isOneOf(InputType.COLUMN));
            }

        }

        static final class IsImageValue extends CompositePredicateProvider {

            IsImageValue() {
                super(i -> i.getEnum(ContentTypeRef.class).isOneOf(ContentType.IMAGE));
            }

        }

        Contents(final String textValue, final String imageColumn) {
            m_textValue = textValue;
            m_imageColumn = imageColumn;
        }

        Contents() {
            this("", "");
        }

        static final class ImageCellColumns extends CompatibleColumnsProvider {

            protected ImageCellColumns() {
                super(PNGImageValue.class);
            }

        }

    }

    static final class TextCellColumns extends CompatibleColumnsProvider {

        protected TextCellColumns() {
            super(StringValue.class);
        }

    }

}
