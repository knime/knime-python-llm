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
 *   6 Feb 2020 (Marc Bux, KNIME GmbH, Berlin, Germany): created
 */
package org.knime.python.llm.java.writer;

import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.context.NodeCreationConfiguration;
import org.knime.core.node.defaultnodesettings.SettingsModelBoolean;
import org.knime.core.node.defaultnodesettings.SettingsModelString;
import org.knime.filehandling.core.connections.FSCategory;
import org.knime.filehandling.core.defaultnodesettings.filechooser.writer.FileOverwritePolicy;
import org.knime.filehandling.core.node.portobject.SelectionMode;
import org.knime.filehandling.core.node.portobject.writer.PortObjectWriterNodeConfig;

/**
 *
 * @author Marc Bux, KNIME GmbH, Berlin, Germany
 */
final class WorkflowWriterNodeConfig extends PortObjectWriterNodeConfig {

    private static final SelectionMode SELECTION_MODE = SelectionMode.FOLDER;

    private static final String CFG_EXISTS_OPTION = "exists";

    private final SettingsModelString m_existsOption =
        new SettingsModelString(CFG_EXISTS_OPTION, WorkflowWriterNodeDialog.EXISTS_OPTION_DEF.getActionCommand());

    private static final String CFG_ARCHIVE = "archive";

    private final SettingsModelBoolean m_archive = new SettingsModelBoolean(CFG_ARCHIVE, false);

    private static final String CFG_OPEN_AFTER_WRITE = "open";

    private final SettingsModelBoolean m_openAfterWrite = new SettingsModelBoolean(CFG_OPEN_AFTER_WRITE, false);

    private static final String CFG_USE_CUSTOM_NAME = "use-custom-name";

    private final SettingsModelBoolean m_useCustomName = new SettingsModelBoolean(CFG_USE_CUSTOM_NAME, false);

    private static final String CUSTOM_NAME = "custom-name";

    private final SettingsModelString m_customName = new SettingsModelString(CUSTOM_NAME, "workflow");

    private static final String CFG_IO_NODES = "io-nodes";

    private final SettingsModelIONodes m_ioNodes = new SettingsModelIONodes(CFG_IO_NODES);

    private static final String CFG_DO_REMOVE_LINKS = "do_remove_template_links";

    private static final boolean DO_REMOVE_LINKS_DEFAULT = false;

    /**
     * Whether to remove links of linked metanodes and components before writing the workflow segment.
     * @since 4.5
     */
    private final SettingsModelBoolean m_doRemoveTemplateLinks = new SettingsModelBoolean(CFG_DO_REMOVE_LINKS,
            DO_REMOVE_LINKS_DEFAULT);


    private static final String CFG_DO_UPDATE_LINKS = "do_update_template_links";

    private static final boolean DO_UPDATE_LINKS_DEFAULT = false;

    /**
     * Whether to update linked metanodes and components before writing the workflow segment.
     * If this is enabled and links are also set to be removed, the metanodes/components will first be
     * updated and then disconnected.
     * @since 4.5
     */
    private final SettingsModelBoolean m_doUpdateTemplateLinks = new SettingsModelBoolean(CFG_DO_UPDATE_LINKS,
            DO_UPDATE_LINKS_DEFAULT);





    /**
     * Constructor for configs in which the file chooser doesn't filter on file suffixes.
     *
     * @param creationConfig {@link NodeCreationConfiguration} of the corresponding KNIME node
     */
    WorkflowWriterNodeConfig(final NodeCreationConfiguration creationConfig) {
        super(builder(creationConfig).withSelectionMode(SELECTION_MODE)
            .withFileOverwritePolicies(FileOverwritePolicy.APPEND)
            .withConvenienceFS(FSCategory.LOCAL, FSCategory.HUB_SPACE, FSCategory.MOUNTPOINT, FSCategory.RELATIVE));
    }

    SettingsModelString getExistsOption() {
        return m_existsOption;
    }

    SettingsModelBoolean isUseCustomName() {
        return m_useCustomName;
    }

    SettingsModelString getCustomName() {
        return m_customName;
    }

    SettingsModelBoolean isArchive() {
        return m_archive;
    }

    SettingsModelBoolean isOpenAfterWrite() {
        return m_openAfterWrite;
    }

    SettingsModelIONodes getIONodes() {
        return m_ioNodes;
    }

    SettingsModelBoolean getDoRemoveTemplateLinks() {
        return m_doRemoveTemplateLinks;
    }

    SettingsModelBoolean getDoUpdateTemplateLinks() {
        return m_doUpdateTemplateLinks;
    }

    @Override
    protected void validateConfigurationForModel(final NodeSettingsRO settings) throws InvalidSettingsException {
        super.validateConfigurationForModel(settings);
        m_existsOption.validateSettings(settings);
        m_useCustomName.validateSettings(settings);
        m_customName.validateSettings(settings);
        m_archive.validateSettings(settings);
        m_openAfterWrite.validateSettings(settings);
        m_ioNodes.validateSettings(settings);
        if (settings.containsKey(CFG_DO_REMOVE_LINKS)) {
            m_doRemoveTemplateLinks.validateSettings(settings);
        }
        if (settings.containsKey(CFG_DO_UPDATE_LINKS)) {
            m_doUpdateTemplateLinks.validateSettings(settings);
        }
    }

    @Override
    protected void saveConfigurationForModel(final NodeSettingsWO settings) {
        super.saveConfigurationForModel(settings);
        m_existsOption.saveSettingsTo(settings);
        m_useCustomName.saveSettingsTo(settings);
        m_customName.saveSettingsTo(settings);
        m_archive.saveSettingsTo(settings);
        m_openAfterWrite.saveSettingsTo(settings);
        m_ioNodes.saveSettingsTo(settings);
        m_doRemoveTemplateLinks.saveSettingsTo(settings);
        m_doUpdateTemplateLinks.saveSettingsTo(settings);
    }

    @Override
    protected void loadConfigurationForModel(final NodeSettingsRO settings) throws InvalidSettingsException {
        super.loadConfigurationForModel(settings);
        m_existsOption.loadSettingsFrom(settings);
        m_useCustomName.loadSettingsFrom(settings);
        m_customName.loadSettingsFrom(settings);
        m_archive.loadSettingsFrom(settings);
        m_openAfterWrite.loadSettingsFrom(settings);
        m_ioNodes.loadSettingsFrom(settings);
        if (settings.containsKey(CFG_DO_REMOVE_LINKS)) {
            // backwards compatibility: load default value if setting not present.
            m_doRemoveTemplateLinks.loadSettingsFrom(settings);
        } else {
            m_doRemoveTemplateLinks.setBooleanValue(DO_REMOVE_LINKS_DEFAULT);
        }
        if (settings.containsKey(CFG_DO_UPDATE_LINKS)) {
            m_doUpdateTemplateLinks.loadSettingsFrom(settings);
        } else {
            m_doUpdateTemplateLinks.setBooleanValue(DO_UPDATE_LINKS_DEFAULT);
        }
    }

}
