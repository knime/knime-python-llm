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

import org.knime.ai.core.data.message.MessageValue.MessageType;
import org.knime.ai.core.node.message.SettingsUtils.ColumnProviders.JsonColumns;
import org.knime.ai.core.node.message.SettingsUtils.ColumnProviders.PngColumns;
import org.knime.ai.core.node.message.SettingsUtils.ColumnProviders.StringColumns;
import org.knime.ai.core.node.message.SettingsUtils.CompositePredicateProvider;
import org.knime.node.parameters.NodeParameters;
import org.knime.core.webui.node.dialog.defaultdialog.setting.singleselection.NoneChoice;
import org.knime.core.webui.node.dialog.defaultdialog.setting.singleselection.StringOrEnum;
import org.knime.node.parameters.Widget;
import org.knime.node.parameters.array.ArrayWidget;
import org.knime.node.parameters.layout.After;
import org.knime.node.parameters.layout.HorizontalLayout;
import org.knime.node.parameters.layout.Layout;
import org.knime.node.parameters.layout.Section;
import org.knime.node.parameters.updates.Effect;
import org.knime.node.parameters.updates.EffectPredicate;
import org.knime.node.parameters.updates.ParameterReference;
import org.knime.node.parameters.updates.ValueReference;
import org.knime.node.parameters.updates.Effect.EffectType;
import org.knime.node.parameters.widget.choices.ChoicesProvider;
import org.knime.node.parameters.widget.choices.Label;
import org.knime.node.parameters.widget.choices.ValueSwitchWidget;
import org.knime.node.parameters.widget.text.TextAreaWidget;

/**
 * Settings for the Message Creator node.
 *
 * @author Seray Arslan, KNIME GmbH, Konstanz, Germany
 */
@SuppressWarnings("restriction")
final class MessageCreatorNodeSettings implements NodeParameters {

    enum InputType {
            @Label("Value")
            VALUE,

            @Label("Column")
            COLUMN;
    }

    @Section(title = "Sender configuration")
    interface SenderConfigurationLayout {
    }

    @Section(title = "Content")
    @After(SenderConfigurationLayout.class)
    interface ContentLayout {
    }

    @Section(title = "Tool calls")
    @After(ContentLayout.class)
    @Effect(predicate = ShouldShowToolCallsSection.class, type = EffectType.SHOW)
    interface ToolCallsLayout {
    }

    @Section(title = "Output")
    @After(ToolCallsLayout.class)
    interface OutputLayout {
    }

    static final class RoleInputTypeRef implements ParameterReference<InputType> {
    }

    static final class RoleTypeRef implements ParameterReference<MessageType> {
    }

    private static final class IsValueRoleInputType extends CompositePredicateProvider {
        IsValueRoleInputType() {
            super(i -> i.getEnum(RoleInputTypeRef.class).isOneOf(InputType.VALUE));
        }
    }

    private static final class IsColumnRoleInputType extends CompositePredicateProvider {
        IsColumnRoleInputType() {
            super(i -> i.getEnum(RoleInputTypeRef.class).isOneOf(InputType.COLUMN));
        }
    }

    private static final class IsToolRole extends CompositePredicateProvider {
        IsToolRole() {
            super(i -> i.getEnum(RoleTypeRef.class).isOneOf(MessageType.TOOL));
        }
    }

    private static final class IsAIRole extends CompositePredicateProvider {
        IsAIRole() {
            super(i -> i.getEnum(RoleTypeRef.class).isOneOf(MessageType.AI));
        }
    }

    private static final class IsValueInputAndAIRole extends CompositePredicateProvider {
        IsValueInputAndAIRole() {
            super(new IsValueRoleInputType(), new IsAIRole());
        }
    }

    private static final class ShouldShowToolCallsSection extends CompositePredicateProvider {
        ShouldShowToolCallsSection() {
            super(EffectPredicate::or, new IsColumnRoleInputType(), new IsValueInputAndAIRole());
        }
    }

    private static final class ShowWhenColumnInputOrToolRole extends CompositePredicateProvider {
        ShowWhenColumnInputOrToolRole() {
            super(EffectPredicate::or, new IsToolRole(), new IsColumnRoleInputType());
        }
    }

    @Layout(SenderConfigurationLayout.class)
    @Widget(title = "Role input",
        description = "Choose whether the message role is provided as a value or as a column.")
    @ValueSwitchWidget
    @ValueReference(RoleInputTypeRef.class)
    InputType m_roleInputType = InputType.VALUE;

    @Layout(SenderConfigurationLayout.class)
    @Widget(title = "Role", description = "Specify the role of the message, such as User, AI or Tool.")
    @Effect(predicate = IsValueRoleInputType.class, type = EffectType.SHOW)
    @ValueReference(RoleTypeRef.class)
    MessageType m_roleValue = MessageType.USER;

    @Layout(SenderConfigurationLayout.class)
    @Widget(title = "Role column",
        description = "Select the column from the input table that contains the message roles.")
    @Effect(predicate = IsValueRoleInputType.class, type = EffectType.HIDE)
    @ChoicesProvider(StringColumns.class)
    String m_roleColumn;

    @Layout(SenderConfigurationLayout.class)
    @Widget(title = "Name",
        description = "(Optional) Select the input table column containing the name, or leave as None. "
            + "Currently only used for the Tool role.\n"
            + "<ul>"
            + "<li><b>Purpose:</b> This column provides a name to differentiate between participants of the same role. "
            + "In conversations where multiple entities share the same role (such as several users or multiple tools), "
            + "the name property allows you to uniquely identify each participant.</li>"
            + "<li><b>Usage:</b> This is especially useful when simulating structured multi-user dialogues or integrating multiple tools, "
            + "as it lets the model track who said what and maintain context across turns.</li>"
            + "<li><b>Example:</b> In a chat where two users are involved, assigning names like <code>user_A</code> and <code>user_B</code> "
            + "helps the model respond accurately and refer to the right person or tool, even though both technically have the same role.</li>"
            + "</ul>")
    @Effect(predicate = ShowWhenColumnInputOrToolRole.class, type = EffectType.SHOW)
    @ChoicesProvider(StringColumns.class)
    StringOrEnum<NoneChoice> m_nameColumn = new StringOrEnum<>(NoneChoice.NONE);

    @Layout(SenderConfigurationLayout.class)
    @Widget(title = "Tool call ID column",
        description = "(Optional) Select the column containing tool call IDs. The ID should match "
            + "that of the tool call generated by an agent.")
    @Effect(predicate = ShowWhenColumnInputOrToolRole.class, type = EffectType.SHOW)
    @ChoicesProvider(StringColumns.class)
    StringOrEnum<NoneChoice> m_toolCallIdColumn = new StringOrEnum<>(NoneChoice.NONE);

    @Layout(ContentLayout.class)
    @Widget(title = "Message content",
        description = "Define the content of the message, including text and/or image parts.")
    @ArrayWidget(addButtonText = "Add content part", showSortButtons = false, elementTitle = "Content part")
    Contents[] m_content = {new Contents()};

    @Layout(ToolCallsLayout.class)
    @Widget(title = "Tool calls",
        description = "Define tool calls associated with the message, including tool names, IDs, and arguments.")
    @ArrayWidget(addButtonText = "Add tool call", showSortButtons = false, elementTitle = "Tool Call")
    ToolCallSettings[] m_toolCalls = {};

    @Layout(OutputLayout.class)
    @Widget(title = "Message column name",
        description = "Name of the output column that will contain the created messages.")
    String m_messageColumnName = "Message";

    @Layout(OutputLayout.class)
    @Widget(title = "Remove input columns",
        description = "If selected, the columns used to create the new message will be removed from the output table.")
    boolean m_removeInputColumns = false;

    static final class ToolCallSettings implements NodeParameters {
        @Widget(title = "Tool name column", description = "Select the input table column containing the tool name.")
        @ChoicesProvider(StringColumns.class)
        String m_toolNameColumn;

        @Widget(title = "Tool call ID column", description = "Select the input table column containing the tool IDs.")
        @ChoicesProvider(StringColumns.class)
        String m_toolIdColumn;

        @Widget(title = "Arguments column",
            description = "Select the input table column containing the tool arguments (JSON string).")
        @ChoicesProvider(JsonColumns.class)
        String m_argumentsColumn;
    }

    static final class Contents implements NodeParameters {

        enum ContentType {
                @Label("Text")
                TEXT,

                @Label("Image")
                IMAGE;
        }

        @HorizontalLayout
        interface ContentsLayout {
        }

        @After(ContentsLayout.class)
        interface ContentsValueLayout {
        }

        @Layout(ContentsLayout.class)
        @Widget(title = "Type", description = "Select the type of the message content (Text or Image).")
        @ValueSwitchWidget
        @ValueReference(ContentTypeRef.class)
        ContentType m_contentType = ContentType.TEXT;

        @Layout(ContentsLayout.class)
        @Widget(title = "Input type",
            description = "Specify how the content is provided: either directly as a value or from a column in the input table.")
        @ValueSwitchWidget
        @ValueReference(InputTypeRef.class)
        @Effect(predicate = HasInputTypes.class, type = EffectType.SHOW)
        InputType m_inputType = InputType.VALUE;

        @Layout(ContentsValueLayout.class)
        @Widget(title = "Image column",
            description = "Select the column from the input table that contains PNG images.")
        @ChoicesProvider(PngColumns.class)
        @Effect(predicate = IsImageValue.class, type = EffectType.SHOW)
        String m_imageColumn;

        @Layout(ContentsValueLayout.class)
        @Widget(title = "Text value", description = "Enter the text content.")
        @TextAreaWidget
        @Effect(predicate = IsTextValue.class, type = EffectType.SHOW)
        String m_textValue;

        @Layout(ContentsValueLayout.class)
        @Widget(title = "Text column",
            description = "Select the column from the input table that contains text values.")
        @ChoicesProvider(StringColumns.class)
        @Effect(predicate = IsTextColumn.class, type = EffectType.SHOW)
        String m_textColumn;

        static final class ContentTypeRef implements ParameterReference<ContentType> {
        }

        static final class InputTypeRef implements ParameterReference<InputType> {
        }

        static final class HasInputTypes extends CompositePredicateProvider {
            HasInputTypes() {
                super(i -> i.getEnum(ContentTypeRef.class).isOneOf(ContentType.TEXT));
            }
        }

        static final class IsTextValue extends CompositePredicateProvider {
            IsTextValue() {
                super(EffectPredicate::and, i -> i.getEnum(ContentTypeRef.class).isOneOf(ContentType.TEXT),
                    i -> i.getEnum(InputTypeRef.class).isOneOf(InputType.VALUE));
            }
        }

        static final class IsTextColumn extends CompositePredicateProvider {
            IsTextColumn() {
                super(EffectPredicate::and, i -> i.getEnum(ContentTypeRef.class).isOneOf(ContentType.TEXT),
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
    }
}
