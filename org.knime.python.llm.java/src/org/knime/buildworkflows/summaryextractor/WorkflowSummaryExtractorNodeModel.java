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
 *   Mar 4, 2020 (hornm): created
 */
package org.knime.python.llm.java.summaryextractor;

import static org.knime.core.util.workflowsummary.WorkflowSummaryUtil.writeJSON;
import static org.knime.core.util.workflowsummary.WorkflowSummaryUtil.writeXML;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Set;

import org.knime.core.data.DataTableSpec;
import org.knime.core.data.DataType;
import org.knime.core.data.def.DefaultRow;
import org.knime.core.data.json.JSONCell;
import org.knime.core.data.json.JSONCellFactory;
import org.knime.core.data.xml.XMLCell;
import org.knime.core.data.xml.XMLCellFactory;
import org.knime.core.node.BufferedDataContainer;
import org.knime.core.node.BufferedDataTable;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.ExecutionMonitor;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeLogger;
import org.knime.core.node.NodeModel;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.defaultnodesettings.SettingsModelBoolean;
import org.knime.core.node.defaultnodesettings.SettingsModelString;
import org.knime.core.node.port.PortObject;
import org.knime.core.node.port.PortObjectSpec;
import org.knime.core.node.port.PortType;
import org.knime.core.node.workflow.WorkflowLoadHelper;
import org.knime.core.node.workflow.WorkflowManager;
import org.knime.core.node.workflow.capture.BuildWorkflowsUtil;
import org.knime.core.node.workflow.capture.WorkflowPortObject;
import org.knime.core.node.workflow.capture.WorkflowPortObjectSpec;
import org.knime.core.node.workflow.capture.WorkflowSegment;
import org.knime.core.util.UniqueNameGenerator;
import org.knime.core.util.workflowsummary.WorkflowSummary;
import org.knime.core.util.workflowsummary.WorkflowSummaryCreator;

import com.knime.enterprise.utility.jackson.ObjectMapperUtil;

/**
 * Node to extract WorkflowSummaries.
 *
 * @author Gabriel Einsdorf, KNIME GmbH, Konstanz, Germany
 * @author Martin Horn, KNIME GmbH, Konstanz, Germany
 * @author Jannik Löscher, KNIME GmbH, Konstanz, Germany
 */
final class WorkflowSummaryExtractorNodeModel extends NodeModel {

    private static final NodeLogger LOGGER = NodeLogger.getLogger(WorkflowSummaryExtractorNodeModel.class);

    static final String FMT_SELECTION_JSON = "JSON";

    static final String FMT_SELECTION_XML = "XML";

    private static final String COLUMN_NAME = "workflow summary";

    private static final String ROW_NAME = "summary";

    private static final String METADATA_COL_NAME = "metadata";

    static SettingsModelString createOutputFormatSelectionModel() {
        return new SettingsModelString("output_format", FMT_SELECTION_JSON);
    }

    static SettingsModelString createColumnNameModel() {
        return new SettingsModelString("column_name", COLUMN_NAME);
    }

    static SettingsModelBoolean createOutputMetadataModel() {
        return new SettingsModelBoolean("output_metadata", false);
    }

    static SettingsModelBoolean createCheckForUpdatesModel() {
        return new SettingsModelBoolean("check_for_updates", false);
    }

    private final SettingsModelString m_outputFormat = createOutputFormatSelectionModel();

    private final SettingsModelBoolean m_outputMetadata = createOutputMetadataModel();

    private final SettingsModelString m_columnName = createColumnNameModel();

    /**
     * Added with AP-19535: Whether to check for updates in the workflow
     */
    private final SettingsModelBoolean m_checkForUpdates = createCheckForUpdatesModel();

    WorkflowSummaryExtractorNodeModel() {
        super(new PortType[]{WorkflowPortObject.TYPE}, new PortType[]{BufferedDataTable.TYPE});
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected DataTableSpec[] configure(final PortObjectSpec[] inSpecs) {
        return new DataTableSpec[]{createSpec()};
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected PortObject[] execute(final PortObject[] inObjects, final ExecutionContext exec) throws Exception {
        final WorkflowPortObjectSpec spec = ((WorkflowPortObject)inObjects[0]).getSpec();
        final WorkflowSegment segment = spec.getWorkflowSegment();
        try {
            final WorkflowManager wfm = BuildWorkflowsUtil.loadWorkflow(segment, this::setWarningMessage);
            if (wfm != null) {
                // Added with AP-19535: Set the UpdateStatus of linked templates (without updating the node)
                if (m_checkForUpdates.getBooleanValue()) {
                    checkForMetanodeUpdates(wfm);
                }
                return new BufferedDataTable[]{fillTable(exec, wfm)};
            } else {
                throw new IllegalStateException("No workflow context available");
            }
        } finally {
            segment.disposeWorkflow();
        }
    }

    /**
     * Added with AP-19535: Checks all linked metanodes/components for updates
     *
     * @param wfm The workflow in which to operate
     */
    private void checkForMetanodeUpdates(final WorkflowManager wfm) {
        var errors = false;
        for (var nodeID : wfm.getLinkedMetaNodes(true)) { // check every template node recursively
            try {
                var helper = new WorkflowLoadHelper(true, wfm.getContextV2());
                wfm.checkUpdateMetaNodeLink(nodeID, helper);
            } catch (IOException e) {
                errors = true;
                var node = wfm.findNodeContainer(nodeID);
                LOGGER.warn("Error checking for updates of \"" + node.getNameWithID()
                    + "\". Maybe the template has been moved or the resource is not mounted?", e);
            }
        }
        if (errors) {
            setWarningMessage("There have been errors checking for some metanode / component updates");
        }
    }

    private DataTableSpec createSpec() {
        final var outputFormat = isJsonSelected() ? JSONCell.TYPE : XMLCell.TYPE;
        final var wfSummaryCol = m_columnName.getStringValue();
        if (m_outputMetadata.getBooleanValue()) {
            return new DataTableSpec(new String[]{wfSummaryCol,
                new UniqueNameGenerator(Set.of(wfSummaryCol)).newName(METADATA_COL_NAME)},
                new DataType[]{ outputFormat, outputFormat });
        }
        return new DataTableSpec(new String[]{ wfSummaryCol }, new DataType[]{ outputFormat });
    }

    private BufferedDataTable fillTable(final ExecutionContext exec, final WorkflowManager wfm)
        throws Exception {
        final WorkflowSummary summary = WorkflowSummaryCreator.create(wfm, false, Collections.emptyList());
        final var metadata = wfm.getMetadata();
        final BufferedDataContainer container = exec.createDataContainer(createSpec());
        try (ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            final var outputMetadata = m_outputMetadata.getBooleanValue();
            final var utf8 = StandardCharsets.UTF_8.name();
            if (isJsonSelected()) {
                writeJSON(out, summary, false);
                final var wfSummaryJson = JSONCellFactory.create(out.toString(utf8), false);
                if (outputMetadata) {
                    final var metaJson = ObjectMapperUtil.getInstance().getObjectMapper().writeValueAsString(metadata);
                    container.addRowToTable(new DefaultRow(ROW_NAME, wfSummaryJson,
                        JSONCellFactory.create(metaJson, false)));
                } else {
                    container.addRowToTable(new DefaultRow(ROW_NAME, wfSummaryJson));
                }
            } else {
                writeXML(out, summary, false);
                final var wfSummaryXml = XMLCellFactory.create(out.toString(utf8));
                if (outputMetadata) {
                    final var metaXml = metadata.toXML();
                    container.addRowToTable(new DefaultRow(ROW_NAME, wfSummaryXml, XMLCellFactory.create(metaXml)));
                } else {
                    container.addRowToTable(new DefaultRow(ROW_NAME, wfSummaryXml));
                }
            }

            container.close();
            return container.getTable();
        }
    }

    private boolean isJsonSelected() {
        return m_outputFormat.getStringValue().equals(FMT_SELECTION_JSON);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void loadInternals(final File nodeInternDir, final ExecutionMonitor exec) {
        //
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void saveInternals(final File nodeInternDir, final ExecutionMonitor exec) {
        //
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void saveSettingsTo(final NodeSettingsWO settings) {
        m_outputFormat.saveSettingsTo(settings);
        m_columnName.saveSettingsTo(settings);
        m_checkForUpdates.saveSettingsTo(settings);
        m_outputMetadata.saveSettingsTo(settings);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void validateSettings(final NodeSettingsRO settings) throws InvalidSettingsException {
        m_outputFormat.validateSettings(settings);
        m_columnName.validateSettings(settings);
        // Added with AP-19535
        if (settings.containsKey(m_checkForUpdates.getConfigName())) {
            m_checkForUpdates.validateSettings(settings);
        }
        // added in 5.1
        if (settings.containsKey(m_outputMetadata.getConfigName())) {
            m_outputMetadata.validateSettings(settings);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void loadValidatedSettingsFrom(final NodeSettingsRO settings) throws InvalidSettingsException {
        m_outputFormat.loadSettingsFrom(settings);
        m_columnName.loadSettingsFrom(settings);
        // Added with AP-19535
        if (settings.containsKey(m_checkForUpdates.getConfigName())) {
            m_checkForUpdates.loadSettingsFrom(settings);
        }
        // added in 5.1
        if (settings.containsKey(m_outputMetadata.getConfigName())) {
            m_outputMetadata.loadSettingsFrom(settings);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void reset() {
        //
    }

}
