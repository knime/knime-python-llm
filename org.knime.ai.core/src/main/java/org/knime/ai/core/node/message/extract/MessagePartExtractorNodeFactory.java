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
 *   May 30, 2025 (Adrian Nembach, KNIME GmbH, Konstanz, Germany): created
 */
package org.knime.ai.core.node.message.extract;

import org.knime.core.webui.node.impl.WebUINodeConfiguration;
import org.knime.core.webui.node.impl.WebUINodeFactory;

/**
 * Factory for the Message Part Extractor node, which extracts parts from messages in a table.
 *
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 */
@SuppressWarnings("restriction")
public final class MessagePartExtractorNodeFactory extends WebUINodeFactory<MessagePartExtractorNodeModel> {

    private static final WebUINodeConfiguration CONFIG = WebUINodeConfiguration.builder()
            .name("Message Part Extractor")
            .icon("./Message-part-extractor.png")
            .shortDescription("Extracts specific parts from messages in a table.")
            .fullDescription("""
                <p>
                    This node extracts specific parts from messages in a table, allowing you to work with content types such as text, images, roles, names, and tool calls.
                    The input table is processed row by row, extracting the specified parts from each message.
                </p>
                <p>
                    Extracted parts:
                </p>
                <ul>
                    <li><b>Role</b>: Extracts the role of the message, such as User, AI, or Tool.</li>
                    <li><b>Name</b>: Extracts the name associated with the message.</li>
                    <li><b>Text Content</b>: Extracts text content from the message.</li>
                    <li><b>Image Content</b>: Extracts image content from the message.</li>
                    <li><b>Tool Calls</b>: Extracts tool calls, including tool names, IDs, and arguments.</li>
                    <li><b>Tool Call ID</b>: Extracts the ID of the tool call associated with the message.</li>
                </ul>
            """)
            .modelSettingsClass(MessagePartExtractorSettings.class)
            .addInputTable("Message table", "Table containing messages to extract parts from.")
            .addOutputTable("Message part table", "Table with extracted message parts.")
            .build();

    /**
     * Constructor for the Message Part Extractor node factory.
     */
    public MessagePartExtractorNodeFactory() {
        super(CONFIG);
    }

    @Override
    public MessagePartExtractorNodeModel createNodeModel() {
        return new MessagePartExtractorNodeModel();
    }

}
