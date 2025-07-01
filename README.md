# ![Image](https://www.knime.com/sites/default/files/knime_logo_github_40x40_4layers.png) KNIME® - AI Extension

This repository is maintained by [KNIME Team spAIceship](mailto:team-spaiceship@knime.com).

## Content

This repository contains the source code for the KNIME AI Extension. The extension offers nodes for building LLM-powered no-code workflows, and supports interfacing with hosted services, as well as locally-running models.

## Development Notes

### Resources
- Refer to the [Python extension development documentation](https://docs.knime.com/latest/pure_python_node_extensions_guide/index.html#introduction) for building pure-Python nodes for KNIME.
- For working with this codebase or developing KNIME Analytics Platform extensions, see the _knime-sdk-setup_ repository on [BitBucket](https://bitbucket.org/KNIME/knime-sdk-setup) or [GitHub](http://github.com/knime/knime-sdk-setup).

### Code Formatting
- Use the [Ruff Formatter](https://docs.astral.sh/ruff/formatter/) to ensure consistent Python code style:
  ```bash
  ruff format .
  ```

### Development Environment Setup
1. **Install dependencies** using [Pixi](https://pixi.sh/):
   ```bash
   pixi install -e dev
   ```
2. **Ensure Java plugins are present**:  
  The KNIME AI Extension now includes both Python and Java components. To use the extension, you must have the required Java plugins available in your KNIME Analytics Platform. You can achieve this in one of two ways:
  - **Option 1: Install the AI Extension**  
    Install the AI Extension directly into your KNIME Analytics Platform using the standard update sites. This will ensure all necessary Java plugins are installed.
  - **Option 2: Develop with Eclipse**  
    If you are developing or modifying the Java plugins, check out the relevant plugin projects in Eclipse and launch KNIME from your Eclipse workspace. This allows you to work with the latest source code for both Java and Python components.

3. **Register the Python extension** in KNIME Analytics Platform:
  - Install the [KNIME Python Extension Development](https://hub.knime.com/knime/extensions/org.knime.features.python3.nodes/latest) feature. It might already be installed if you have any Python-based extension.
  - Create or update your Python `config.yaml` with:
    ```yaml
    org.knime.python.llm:
     src: "<path/to/this/repository>"
     conda_env_path: "<path/to/this/repository>/.pixi/envs/dev"
     debug_mode: true  # Disables Python process caching for live code reloads (slower on Windows)
    ```
  - Point KNIME to your config by adding to `knime.ini`:
    ```
    -Dknime.python.extension.config=<path/to/your/config.yaml>
    ```
  - **Note for Eclipse development:**  
    When developing from Eclipse, the plugin `org.knime.python3.nodes` from the [knime-python](https://bitbucket.org/KNIME/knime-python) repository needs to be checked out or added to the target platform, as it is not included by default.

### Configuring the dev Environment in VS Code
To use the `dev` environment in VS Code for linting, formatting, and debugging:

**Select the Python Interpreter:**
   - Open the Command Palette (Ctrl+Shift+P) and run `Python: Select Interpreter`.
   - Choose the interpreter from `.pixi\envs\dev` (on Windows) or `.pixi/envs/dev/bin/python` (on Linux/macOS).
   - If you do not see the environment, click "Enter interpreter path..." and browse to the Python executable in the `dev` environment.

### Debugging
- Use [`debugpy`](https://github.com/microsoft/debugpy) to enable remote debugging:
  ```python
  import debugpy
  debugpy.listen(5678)
  print("Waiting for debugger attach")
  debugpy.wait_for_client()
  ```
  - Place this at the start of a node's `execute` method for best results (KNIME may start multiple processes; only one can attach).
  - To set breakpoints in code:
    ```python
    import debugpy  # if not already imported
    debugpy.breakpoint()
    ```
- In VS Code, open the Run & Debug view (Ctrl+Shift+D), select "Remote Attach" (localhost:5678), and start debugging. Ensure a Python file is open to see Python debug options.
- For a better experience, add this to your `.vscode/launch.json`:
  ```json
  {
    "version": "0.2.0",
    "configurations": [
      {
        "name": "Python: Remote Attach",
        "type": "python",
        "request": "attach",
        "connect": { "host": "localhost", "port": 5678 },
        "justMyCode": false
      }
    ]
  }
  ```
  - Setting `justMyCode` to `false` allows stepping into library code

### Debugging with the Debugpy Attacher Extension
- For a streamlined debugging experience, you can use the [Debugpy Attacher](https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy-adapter) VS Code extension.
  - **Install the extension:**
    - Open the Extensions view in VS Code (Ctrl+Shift+X) and search for "Debugpy Attacher".
    - Click Install.
  - **Benefits:**
    - No need to manually enter host/port or edit configuration files.
    - Quickly attach to any debugpy-enabled process from within VS Code.
  - **Usage:**
    - Refer the [documentation](https://marketplace.visualstudio.com/items?itemName=DebugPyAttacher.debugpy-attacher) for general usage.
    - Note that when running in KNIME, you do not need to start the Python process, since KNIME will do it.

## Join the Community

- [KNIME Forum](https://forum.knime.com/)

## License

The repository is released under the [GPL v3 License](https://www.gnu.org/licenses/gpl-3.0.html). Please refer to `LICENSE.txt`. Refer to `THIRD_PARTY_LICENSES.txt` for licenses corresponding to
third party libraries and frameworks used in this repository.
