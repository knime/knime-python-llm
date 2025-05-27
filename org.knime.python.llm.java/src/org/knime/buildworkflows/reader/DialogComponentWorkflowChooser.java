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
 *   Apr 12, 2021 (hornm): created
 */
package org.knime.python.llm.java.reader;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

import javax.swing.JPanel;

import org.knime.core.node.FlowVariableModel;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.defaultnodesettings.DialogComponentStringSelection;
import org.knime.core.node.defaultnodesettings.SettingsModelString;
import org.knime.core.node.util.FileSystemBrowser.DialogType;
import org.knime.core.util.hub.ItemVersion;
import org.knime.filehandling.core.connections.ItemVersionAware.RepositoryItemVersion;
import org.knime.filehandling.core.connections.workflowaware.WorkflowAware;
import org.knime.filehandling.core.defaultnodesettings.filechooser.AbstractDialogComponentFileChooser;
import org.knime.filehandling.core.defaultnodesettings.filechooser.StatusMessageReporter;
import org.knime.filehandling.core.defaultnodesettings.filechooser.reader.ReadPathAccessor;
import org.knime.filehandling.core.defaultnodesettings.status.PriorityStatusConsumer;
import org.knime.filehandling.core.defaultnodesettings.status.StatusMessage;
import org.knime.filehandling.core.util.GBCBuilder;

/**
 * Lets one choose a workflow. In case of a {@link WorkflowAware} filesystem it's the workflow directory, in all other
 * cases it's a '.knwf'-file.
 *
 * @author Martin Horn, KNIME GmbH, Konstanz, Germany
 */
final class DialogComponentWorkflowChooser extends AbstractDialogComponentFileChooser<SettingsModelWorkflowChooser> {

    private static final String VERSION_SELECTOR_CURRENT_STATE = "Latest edits";
    private static final String VERSION_SELECTOR_LATEST_VERSION = "Latest version";

    private SettingsModelString m_versionModel;

    private DialogComponentStringSelection m_versionSelector;

    private LinkedHashMap<String, ItemVersion> m_availableVersions = defaultMap();

    /**
     * @param model the model backing the dialog component
     * @param historyID id for the workflow selection history
     * @param locationFvm to overwrite the workflow location with a flow variable
     */
    protected DialogComponentWorkflowChooser(final SettingsModelWorkflowChooser model, final String historyID,
        final FlowVariableModel locationFvm) {
        super(model, historyID, DialogType.OPEN_DIALOG, "Read from", locationFvm,
            DialogComponentWorkflowChooser::createStatusMessageReporter);
        getModel().addChangeListener(e -> updateVersionSelector());
        m_versionModel.addChangeListener(e -> {
            getSettingsModel().setItemVersion(m_availableVersions.get(m_versionModel.getStringValue()));
        });
        updateVersionSelector();
    }

    @Override
    protected void addAdditionalComponents(final JPanel panel, final GBCBuilder gbc) {
        m_versionModel = new SettingsModelString("ignore", VERSION_SELECTOR_CURRENT_STATE);
        m_versionSelector =
            new DialogComponentStringSelection(m_versionModel, "Select version", VERSION_SELECTOR_CURRENT_STATE);

        gbc.incX();
        panel.add(m_versionSelector.getComponentPanel(), gbc.build());
        m_versionSelector.getComponentPanel().setVisible(true);
    }

    private void updateVersionSelector() {
        m_versionSelector.getComponentPanel().setVisible(false);
        updateVersions();
        m_versionSelector.replaceListItems(m_availableVersions.keySet(),
            getTitleForVersion(getSettingsModel().getItemVersion()));
        m_versionSelector.getComponentPanel().setVisible(true);
    }

    private String getTitleForVersion(final ItemVersion v) {
        // reverse lookup for available versions
        if (v != null) {
            for (final var entry : m_availableVersions.entrySet()) {
                if (entry.getValue().equals(v)) {
                    return entry.getKey();
                }
            }
        }
        return null;
    }

    @SuppressWarnings("resource")
    private void updateVersions() {
        m_availableVersions = defaultMap();
        if (!getSettingsModel().canCreateConnection()) {
            return;
        }
        // do not close these resources since we only access them but not create them
        // i.e. closing is handled somewhere else
        final var connection = getSettingsModel().getConnection();
        final var fs = connection.getFileSystem();
        final var fspath = fs.getPath(getSettingsModel().getLocation().getPath());
        final var versionAware = fs.getItemVersionAware();
        if (versionAware.isPresent()) {
            try {
                // fetch named versions
                final var hubVersions = versionAware.get().getRepositoryItemVersions(fspath);
                if (!hubVersions.isEmpty()) {
                    final var versions = new LinkedHashMap<String, ItemVersion>();
                    versions.put(VERSION_SELECTOR_CURRENT_STATE, ItemVersion.currentState());
                    versions.put(VERSION_SELECTOR_LATEST_VERSION, ItemVersion.mostRecent());
                    hubVersions.forEach(v -> versions.put(makeTitle(v), ItemVersion.of((int)v.getVersion())));
                    m_availableVersions = versions;
                }
                // If there are no versions, we set no available versions which means there will be no dialog dropdown
            } catch (IOException e) {//NOSONAR
                // This catches exceptions like NoSuchFile / AccessDenied, which are already handled by the
                // AbstractSettingsModelFileChooser, and pretty-printed to the node dialogue there.
                // We simply do not set / show any available versions (what would the user select there, anyways?)
            }
        }
    }

    private static String makeTitle(final RepositoryItemVersion v) {
        return "Version " + v.getVersion() + ": " + v.getTitle();
    }

    private static StatusMessageReporter createStatusMessageReporter(final SettingsModelWorkflowChooser model) {
        return () -> getStatusMessage(model);
    }

    private static StatusMessage getStatusMessage(final SettingsModelWorkflowChooser model)
        throws IOException, InvalidSettingsException {
        if (model.isDataAreaRelativeLocationSelected()) {
            return errorMessage("Data area relative location not supported");
        }
        try (final ReadPathAccessor accessor = model.createReadPathAccessor()) {
            PriorityStatusConsumer consumer = new PriorityStatusConsumer();
            accessor.getFSPaths(consumer);
            Optional<StatusMessage> msg = consumer.get();
            if (msg.isPresent()) {
                return msg.get();
            }
        }
        return emptyMessage();
    }

    private static StatusMessage errorMessage(final String txt) {
        return new StatusMessage() {

            @Override
            public MessageType getType() {
                return MessageType.ERROR;
            }

            @Override
            public String getMessage() {
                return txt;
            }

        };
    }

    private static StatusMessage emptyMessage() {
        return new StatusMessage() {

            @Override
            public MessageType getType() {
                return MessageType.INFO;
            }

            @Override
            public String getMessage() {
                return "";
            }

        };
    }

    private static LinkedHashMap<String, ItemVersion> defaultMap() {
        return new LinkedHashMap<>(Map.of(VERSION_SELECTOR_CURRENT_STATE, ItemVersion.currentState()));
    }

}
