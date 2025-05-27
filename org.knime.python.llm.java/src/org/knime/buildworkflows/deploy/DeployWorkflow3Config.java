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
 *   Nov 22, 2020 (Mark Ortmann, KNIME GmbH, Berlin, Germany): created
 */
package org.knime.python.llm.java.deploy;

import java.util.regex.Matcher;

import org.knime.python.llm.java.ExistsOption;
import org.knime.python.llm.java.writer.SettingsModelIONodes;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.context.ports.PortsConfiguration;
import org.knime.core.node.defaultnodesettings.SettingsModelBoolean;
import org.knime.core.node.defaultnodesettings.SettingsModelString;
import org.knime.core.node.util.CheckUtils;
import org.knime.core.util.FileUtil;
import org.knime.filehandling.core.defaultnodesettings.EnumConfig;
import org.knime.filehandling.core.defaultnodesettings.filechooser.writer.FileOverwritePolicy;
import org.knime.filehandling.core.defaultnodesettings.filechooser.writer.SettingsModelWriterFileChooser;
import org.knime.filehandling.core.defaultnodesettings.filtermode.SettingsModelFilterMode.FilterMode;

/**
 * Configuration of the "Deploy Workflow to Server" node.
 *
 * @author Mark Ortmann, KNIME GmbH, Berlin, Germany
 */
final class DeployWorkflow3Config {

    /** Config for all configuration entries that show up in the settings tab. */
    static final String CFG_SUB_SETTINGS = "settings";

    private static final boolean DEFAULT_USE_CUSTOM_WORKFLOW_NAME = false;

    private static final String CFG_FOLDER = "workflow_group_selection";

    private static final String CFG_WORKFLOW_EXISTS = "if_workflow_exists";

    private static final String CFG_USE_CUSTOM_NAME = "use_custom_workflow_name";

    private static final String CFG_CUSTOM_WORKFLOW_NAME = "custom_workflow_name";

    private static final String CFG_CREATE_SNAPSHOT = "create_snapshot";

    private static final String CFG_SNAPSHOT_MESSAGE = "snapshot_comment";

    private static final String CFG_INPUTS_AND_OUTPUTS = "inputs_and_outputs";

    /**
     * @see DeployWorkflow3Config#m_doRemoveTemplateLinks
     */
    private static final String CFG_DO_REMOVE_LINKS = "do_remove_template_links";

    private static final boolean DO_REMOVE_LINKS_DEFAULT = false;

    /**
     * @see DeployWorkflow3Config#m_doUpdateTemplateLinks
     */
    private static final String CFG_DO_UPDATE_LINKS = "do_update_template_links";

    private static final boolean DO_UPDATE_LINKS_DEFAULT = false;

    private final SettingsModelWriterFileChooser m_fileChooser;

    private final SettingsModelString m_workflowExists;

    private boolean m_useCustomWorkflowName;

    private final SettingsModelString m_customWorkflowName;

    private final SettingsModelBoolean m_createSnapshot;

    private final SettingsModelString m_snapshotName;

    private final SettingsModelIONodes m_ioNodes;

    /**
     * Whether to remove links of linked metanodes and components before deployment.
     * @since 4.5
     */
    private final SettingsModelBoolean m_doRemoveTemplateLinks;

    /**
     * Whether to update linked metanodes and components before deployment.
     * If this is enabled and links are also set to be removed, the metanodes/components will first be
     * updated and then disconnected.
     * @since 4.5
     */
    private final SettingsModelBoolean m_doUpdateTemplateLinks;

    DeployWorkflow3Config(final PortsConfiguration portsConfig) {
        m_fileChooser = new SettingsModelWriterFileChooser(CFG_FOLDER, portsConfig,
            DeployWorkflow3NodeFactory.FS_CONNECT_GRP_ID, EnumConfig.create(FilterMode.FOLDER),
            EnumConfig.create(FileOverwritePolicy.APPEND));
        m_workflowExists = new SettingsModelString(CFG_WORKFLOW_EXISTS, ExistsOption.getDefault().getActionCommand());
        m_useCustomWorkflowName = DEFAULT_USE_CUSTOM_WORKFLOW_NAME;
        m_customWorkflowName = new SettingsModelString(CFG_CUSTOM_WORKFLOW_NAME, "") {

            @Override
            protected void validateSettingsForModel(final NodeSettingsRO settings) throws InvalidSettingsException {
                super.validateSettingsForModel(settings);

                final String customName = settings.getString(getConfigName());
                CheckUtils.checkSetting(!customName.trim().isEmpty(), "Custom workflow name is empty.");
                final Matcher matcher = FileUtil.ILLEGAL_FILENAME_CHARS_PATTERN.matcher(customName);
                if (matcher.find()) {
                    throw new InvalidSettingsException(String.format(
                        "Illegal character in custom workflow name \"%s\" at index %d.", customName, matcher.start()));
                }
            }
        };
        m_customWorkflowName.setEnabled(m_useCustomWorkflowName);

        m_createSnapshot = new SettingsModelBoolean(CFG_CREATE_SNAPSHOT, false);
        m_snapshotName = new SettingsModelString(CFG_SNAPSHOT_MESSAGE, "") {

            @Override
            protected void validateSettingsForModel(final NodeSettingsRO settings) throws InvalidSettingsException {
                super.validateSettingsForModel(settings);

                final String snapshotName = settings.getString(getConfigName());
                CheckUtils.checkSetting(!snapshotName.trim().isEmpty(), "The snapshot comment cannot be empty");
            }
        };
        m_snapshotName.setEnabled(m_createSnapshot.getBooleanValue());

        m_ioNodes = new SettingsModelIONodes(CFG_INPUTS_AND_OUTPUTS);

        m_doRemoveTemplateLinks = new SettingsModelBoolean(CFG_DO_REMOVE_LINKS, DO_REMOVE_LINKS_DEFAULT);

        m_doUpdateTemplateLinks = new SettingsModelBoolean(CFG_DO_UPDATE_LINKS, DO_UPDATE_LINKS_DEFAULT);

    }

    SettingsModelWriterFileChooser getFileChooserModel() {
        return m_fileChooser;
    }

    SettingsModelString getWorkflowExistsModel() {
        return m_workflowExists;
    }

    SettingsModelString getCustomWorkflowNameModel() {
        return m_customWorkflowName;
    }

    SettingsModelBoolean createSnapshotModel() {
        return m_createSnapshot;
    }

    SettingsModelBoolean getDoRemoveTemplateLinksModel() {
        return m_doRemoveTemplateLinks;
    }

    SettingsModelBoolean getDoUpdateTemplateLinksModel() {
        return m_doUpdateTemplateLinks;
    }

    SettingsModelString getSnapshotNameModel() {
        return m_snapshotName;
    }

    SettingsModelIONodes getIOModel() {
        return m_ioNodes;
    }

    boolean useCustomWorkflowName() {
        return m_useCustomWorkflowName;
    }

    void saveSettingsInModel(final NodeSettingsWO settings) {
        final NodeSettingsWO subSettings = settings.addNodeSettings(CFG_SUB_SETTINGS);
        m_fileChooser.saveSettingsTo(subSettings);
        m_workflowExists.saveSettingsTo(subSettings);
        saveUseCustomWorkflowName(subSettings);
        m_customWorkflowName.saveSettingsTo(subSettings);
        m_createSnapshot.saveSettingsTo(subSettings);
        m_doRemoveTemplateLinks.saveSettingsTo(subSettings);
        m_doUpdateTemplateLinks.saveSettingsTo(subSettings);
        m_snapshotName.saveSettingsTo(subSettings);
        m_ioNodes.saveSettingsTo(settings);
    }

    void validateSettingsInModel(final NodeSettingsRO settings) throws InvalidSettingsException {
        final NodeSettingsRO subSettings = settings.getNodeSettings(CFG_SUB_SETTINGS);
        m_fileChooser.validateSettings(subSettings);
        m_workflowExists.validateSettings(subSettings);
        subSettings.containsKey(CFG_USE_CUSTOM_NAME);
        m_customWorkflowName.validateSettings(subSettings);
        m_createSnapshot.validateSettings(subSettings);
        if (subSettings.containsKey(CFG_DO_REMOVE_LINKS)) {
            // backwards compatibility: do not validate if not present
            m_doRemoveTemplateLinks.validateSettings(subSettings);
        }
        if (subSettings.containsKey(CFG_DO_UPDATE_LINKS)) {
            m_doUpdateTemplateLinks.validateSettings(subSettings);
        }
        m_snapshotName.validateSettings(subSettings);
        m_ioNodes.validateSettings(settings);
    }

    void loadSettingsInModel(final NodeSettingsRO settings) throws InvalidSettingsException {
        final NodeSettingsRO subSettings = settings.getNodeSettings(CFG_SUB_SETTINGS);
        m_fileChooser.loadSettingsFrom(subSettings);
        m_workflowExists.loadSettingsFrom(subSettings);
        m_useCustomWorkflowName = subSettings.getBoolean(CFG_USE_CUSTOM_NAME);
        m_customWorkflowName.loadSettingsFrom(subSettings);
        m_createSnapshot.loadSettingsFrom(subSettings);
        if (subSettings.containsKey(CFG_DO_REMOVE_LINKS)) {
            // backwards compatibility: load default value if setting not present.
            m_doRemoveTemplateLinks.loadSettingsFrom(subSettings);
        } else {
            m_doRemoveTemplateLinks.setBooleanValue(DO_REMOVE_LINKS_DEFAULT);
        }
        if (subSettings.containsKey(CFG_DO_UPDATE_LINKS)) {
            m_doUpdateTemplateLinks.loadSettingsFrom(subSettings);
        } else {
            m_doUpdateTemplateLinks.setBooleanValue(DO_UPDATE_LINKS_DEFAULT);
        }
        m_snapshotName.loadSettingsFrom(subSettings);
        m_ioNodes.loadSettingsFrom(settings);
    }

    void loadUseCustomWorkflowNameInDialog(final NodeSettingsRO settings) {
        m_useCustomWorkflowName = settings.getBoolean(CFG_USE_CUSTOM_NAME, DEFAULT_USE_CUSTOM_WORKFLOW_NAME);
    }

    void saveUseCustomWorkflowNameInDialog(final NodeSettingsWO settings, final boolean useCustomWorkflowName) {
        m_useCustomWorkflowName = useCustomWorkflowName;
        saveUseCustomWorkflowName(settings);
    }

    private void saveUseCustomWorkflowName(final NodeSettingsWO settings) {
        settings.addBoolean(CFG_USE_CUSTOM_NAME, useCustomWorkflowName());
    }

    ExistsOption getExistsOption() {
        return ExistsOption.valueOf(m_workflowExists.getStringValue());
    }

}
