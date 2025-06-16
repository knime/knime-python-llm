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

import org.knime.core.webui.node.impl.WebUINodeConfiguration;
import org.knime.core.webui.node.impl.WebUINodeFactory;

/**
 * Factory for the Message Creator node, which creates a KNIME MessageCell
 *
 * @author Seray Arslan, KNIME GmbH, Konstanz, Germany
 */
@SuppressWarnings("restriction")
public final class MessageCreatorNodeFactory extends WebUINodeFactory<MessageCreatorNodeModel>{

    private static final WebUINodeConfiguration CONFIG = WebUINodeConfiguration.builder()
            .name("Message Creator")
            .icon("./Message-creator.png")
            .shortDescription("Creates a Message containing text and/or image data for a specified role.")
            .fullDescription("""
                <p>
                    This node creates Messages, which can include text and/or image content.
                    These messages can be assigned specific roles such as User, AI, or Tool. It is possible
                    to select columns from the input table to generate messages with mixed content, including
                    both text and image data. Additionally, the node supports embedding tool calls within the
                    AI messages. The input table is processed row by row, converting each row into a
                    Message with one or multiple content parts.
                </p>
                <p>
                    Message types:
                </p>
                <ul>
                    <li><b>User Message</b>: Contains text and/or image content provided by the user and is assigned the User role.
                    These messages are used to represent user input.</li>
                    <li><b>AI Message</b>: Contains text and/or image content, assigned the AI role, and may include tool calls.
                    Tool calls specify the tool name, ID, and arguments. These messages are used to represent AI-generated responses
                    or actions, including invoking tools for specific tasks.</li>
                    <li><b>Tool Message</b>: Contains text and/or image content, assigned the Tool role, and includes the tool call ID.
                    These messages represent outputs or responses from tools invoked.</li>
                </ul>
                """)
            .modelSettingsClass(MessageCreatorNodeSettings.class)
            .addInputTable("Input Table", "The table containing rows to process. Each row will be converted into a Message.")
            .addOutputTable("Message Table", "The resulting table containing created messages.")
            .sinceVersion(5, 5, 0)
            .build();

    /**
     * Constructor for the Message Creator node factory.
     */
    public MessageCreatorNodeFactory() {
        super(CONFIG);
    }

    @Override
    public MessageCreatorNodeModel createNodeModel() {
        return new MessageCreatorNodeModel(CONFIG);
    }
}