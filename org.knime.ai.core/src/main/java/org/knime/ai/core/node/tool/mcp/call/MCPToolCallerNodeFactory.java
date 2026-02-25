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
 * derived from each other. Should, however, the interpretation of the
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
 *   Feb 22, 2026 (chaubold): created
 */
package org.knime.ai.core.node.tool.mcp.call;

import java.io.IOException;
import java.io.StringReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.util.UUID;

import org.knime.core.data.DataCell;
import org.knime.core.data.DataColumnSpecCreator;
import org.knime.core.data.DataRow;
import org.knime.core.data.DataTableSpec;
import org.knime.core.data.RowKey;
import org.knime.core.data.def.DefaultRow;
import org.knime.core.data.def.StringCell;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.KNIMEException;
import org.knime.core.node.agentic.tool.ToolCell;
import org.knime.core.node.agentic.tool.ToolType;
import org.knime.core.node.agentic.tool.ToolValue;
import org.knime.node.DefaultModel.ConfigureInput;
import org.knime.node.DefaultModel.ConfigureOutput;
import org.knime.node.DefaultModel.ExecuteInput;
import org.knime.node.DefaultModel.ExecuteOutput;
import org.knime.node.DefaultNode;
import org.knime.node.DefaultNodeFactory;

import jakarta.json.Json;
import jakarta.json.JsonObject;
import jakarta.json.JsonObjectBuilder;
import jakarta.json.JsonReader;
import jakarta.json.JsonValue;

/**
 * NodeFactory for MCP Tool Caller node.
 *
 * @author Carsten Haubold, KNIME GmbH, Konstanz, Germany
 * @since 5.11
 */
@SuppressWarnings("restriction")
public final class MCPToolCallerNodeFactory extends DefaultNodeFactory {

    private static final DefaultNode NODE = DefaultNode.create() //
        .name("MCP Tool Caller") //
        .icon("./default.png") //
        .shortDescription("Executes an MCP (Model Context Protocol) tool by calling a remote MCP server.") //
        .fullDescription("""
                <p>
                This node executes an MCP (Model Context Protocol) tool by calling a remote MCP server via JSON-RPC 2.0.
                </p>
                <p>
                The node takes a table containing tool definitions as input and executes the tool specified by the
                row index. The tool parameters are provided as a JSON object. The result of the tool execution
                is returned as a string in the output table.
                </p>
                <p>
                The input table must contain at least one column compatible with the Tool type, which contains
                the tool definitions including the MCP server URI and tool name.
                </p>
                """) //
        .sinceVersion(5, 11, 0) //
        .ports(p -> p //
            .addInputTable("Tool Table", "Table containing MCP tool definitions. Must have at least one Tool column.") //
            .addOutputTable("Result", "Table containing the result of the tool execution as a string.")) //
        .model(m -> m //
            .parametersClass(MCPToolCallerNodeParameters.class) //
            .configure(MCPToolCallerNodeFactory::configure) //
            .execute(MCPToolCallerNodeFactory::execute)) //
        .keywords("MCP", "Model Context Protocol", "tool", "JSON-RPC", "agent");

    private static final HttpClient HTTP_CLIENT = HttpClient.newBuilder()
        .version(HttpClient.Version.HTTP_1_1)
        .build();

    /**
     * Default constructor for the node factory.
     */
    public MCPToolCallerNodeFactory() {
        super(NODE);
    }

    private static void configure(final ConfigureInput i, final ConfigureOutput o) {
        // Validate that input table has tool column
        var inSpec = i.getInTableSpec(0);
        if (inSpec == null) {
            o.setWarningMessage("Input table specification is not available.");
            return;
        }

        // Check if there's at least one ToolCell column
        boolean hasToolColumn = false;
        for (int colIdx = 0; colIdx < inSpec.getNumColumns(); colIdx++) {
            var colSpec = inSpec.getColumnSpec(colIdx);
            if (colSpec.getType().isCompatible(ToolValue.class)) {
                hasToolColumn = true;
                break;
            }
        }

        if (!hasToolColumn) {
            o.setWarningMessage("Input table must contain at least one Tool column.");
            return;
        }

        // Validate parameters JSON format
        var params = i.<MCPToolCallerNodeParameters>getParameters();
        try {
            parseParametersJson(params.m_parameters);
        } catch (Exception e) {
            o.setWarningMessage("Parameters must be valid JSON: " + e.getMessage());
            return;
        }

        o.setOutSpec(0, createOutputSpec());
    }

    private static void execute(final ExecuteInput i, final ExecuteOutput o) {
        try {
        var params = i.<MCPToolCallerNodeParameters>getParameters();
        var toolTable = i.getInTable(0);

        // Get the selected tool
        ToolCell toolCell = extractToolCell(toolTable, params.m_rowId);

        // Validate it's an MCP tool
        if (toolCell.getToolType() != ToolType.MCP) {
            throw new InvalidSettingsException(
                "Selected tool is not an MCP tool. Tool type: " + toolCell.getToolType());
        }

        // Parse parameters
        var parameters = parseParametersJson(params.m_parameters);

        // Execute the MCP tool
        var result = executeMCPTool(toolCell, parameters, i.getExecutionContext()::setMessage);

        // Create output table
        var container = i.getExecutionContext().createDataContainer(createOutputSpec());
        container.addRowToTable(new DefaultRow(new RowKey("Result"), new StringCell(result)));
        container.close();

        o.setOutData(0, container.getTable());
        }
        catch (InvalidSettingsException | IOException | InterruptedException e) {
            throw new KNIMEException(e.getMessage(), e).toUnchecked();
        }
    }

    private static ToolCell extractToolCell(final org.knime.core.node.BufferedDataTable toolTable, final int rowIndex)
        throws InvalidSettingsException {

        // Find first tool column
        int toolColumnIndex = -1;
        var spec = toolTable.getDataTableSpec();
        for (int colIdx = 0; colIdx < spec.getNumColumns(); colIdx++) {
            if (spec.getColumnSpec(colIdx).getType().isCompatible(ToolValue.class)) {
                toolColumnIndex = colIdx;
                break;
            }
        }

        if (toolColumnIndex < 0) {
            throw new InvalidSettingsException("No tool column found in input table.");
        }

        // Get the row at the specified index
        int currentIndex = 0;
        for (DataRow row : toolTable) {
            if (currentIndex == rowIndex) {
                DataCell cell = row.getCell(toolColumnIndex);
                if (cell.isMissing()) {
                    throw new InvalidSettingsException("Selected row contains a missing tool value.");
                }
                if (!(cell instanceof ToolCell)) {
                    throw new InvalidSettingsException(
                        "Selected cell is not a ToolCell. Type: " + cell.getClass().getName());
                }
                return (ToolCell)cell;
            }
            currentIndex++;
        }

        throw new InvalidSettingsException(
            "Row index " + rowIndex + " not found in table with " + currentIndex + " rows.");
    }

    private static JsonObject parseParametersJson(final String parametersStr) throws InvalidSettingsException {
        try (JsonReader reader = Json.createReader(new StringReader(parametersStr))) {
            JsonValue value = reader.readValue();
            if (value.getValueType() != JsonValue.ValueType.OBJECT) {
                throw new InvalidSettingsException("Parameters must be a JSON object.");
            }
            return value.asJsonObject();
        } catch (Exception e) {
            throw new InvalidSettingsException("Failed to parse parameters JSON: " + e.getMessage(), e);
        }
    }

    private static String executeMCPTool(final ToolCell toolCell, final JsonObject parameters,
        final java.util.function.Consumer<String> setMessage) throws IOException, InterruptedException {

        // TODO: Support authentication for MCP servers.
        // The ToolCell now carries an optional credentialName (see toolCell.getCredentialName()).
        // When non-null, the credential should be resolved from the workflow's credential store
        // and used to set an Authorization header (e.g. Basic Auth) on the HTTP request below.

        // Create JSON-RPC 2.0 request
        var requestId = UUID.randomUUID().toString();
        JsonObjectBuilder requestBuilder = Json.createObjectBuilder()
            .add("jsonrpc", "2.0")
            .add("id", requestId)
            .add("method", "tools/call")
            .add("params", Json.createObjectBuilder()
                .add("name", toolCell.getToolName())
                .add("arguments", parameters));

        var requestJson = requestBuilder.build().toString();

        // Create HTTP request
        var httpRequest = HttpRequest.newBuilder()
            .uri(URI.create(toolCell.getServerUri()))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(requestJson, StandardCharsets.UTF_8))
            .build();

        // Send request
        setMessage.accept("Calling MCP server...");
        HttpResponse<String> httpResponse = HTTP_CLIENT.send(httpRequest, HttpResponse.BodyHandlers.ofString());

        // Check HTTP status
        if (httpResponse.statusCode() != 200) {
            throw new IOException("MCP server returned HTTP " + httpResponse.statusCode() + ": "
                + httpResponse.body());
        }

        // Parse JSON-RPC response
        try (JsonReader reader = Json.createReader(new StringReader(httpResponse.body()))) {
            JsonObject responseJson = reader.readObject();

            // Check for JSON-RPC error
            if (responseJson.containsKey("error")) {
                var error = responseJson.getJsonObject("error");
                throw new IOException("MCP tool execution failed: " + error.toString());
            }

            // Extract result
            if (!responseJson.containsKey("result")) {
                throw new IOException("MCP response missing 'result' field");
            }

            var result = responseJson.get("result");
            return result.toString();

        } catch (Exception e) {
            throw new IOException("Failed to parse MCP response: " + e.getMessage(), e);
        }
    }

    private static DataTableSpec createOutputSpec() {
        return new DataTableSpec(new DataColumnSpecCreator("Result", StringCell.TYPE).createSpec());
    }
}
