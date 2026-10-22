#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat << 'EOF'
Usage: ./run.sh [OPTIONS] <path/to/notebook.ipynb>

Wraps headless Jupyter notebook execution inside a detached tmux session,
sets up and activates the virtual environment (.venv or Conda) with
dependencies from pyproject.toml (without requiring uv), and persists both
the executed notebook results and full execution logs.

Options:
  --venv <path>           Path to virtualenv directory (default: .venv).
  -e, --env <name>        Conda environment name (if using Conda).
  -p, --prefix <path>     Conda environment prefix directory.
  --skip-sync             Skip syncing dependencies from pyproject.toml.
  -o, --output <path>     Path for the executed output notebook.
                          (default: <notebook_dir>/<name>.executed.ipynb)
  -i, --inplace           Overwrite the input notebook in place with results.
  -s, --session <name>    Custom tmux session name.
                          (default: nb_<clean_name>_<timestamp>)
  -l, --log <path>        Path to execution log file.
                          (default: <notebook_dir>/<name>_<timestamp>.run.log)
  -t, --timeout <sec>     Execution timeout per cell in seconds (-1 for none).
                          (default: -1)
  -k, --kernel <name>     Jupyter kernel name (default: python3).
  -a, --attach            Attach to the tmux session immediately after launch.
  -h, --help              Show this help message.

Examples:
  ./run.sh notebooks/explorer.ipynb
  ./run.sh experiments/pendulum-sweeps/parallel_train.ipynb
  ./run.sh --inplace experiments/pendulum-sweeps/parallel_train.ipynb
  ./run.sh --skip-sync -a experiments/pendulum-sweeps/parallel_train.ipynb
EOF
}

NOTEBOOK_ARG=""
OUTPUT_ARG=""
INPLACE=false
SESSION_ARG=""
LOG_ARG=""
TIMEOUT="-1"
KERNEL_ARG=""
ATTACH=false
ENV_NAME_ARG=""
ENV_PREFIX_ARG=""
VENV_ARG=""
SKIP_SYNC=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --venv)
            VENV_ARG="$2"
            shift 2
            ;;
        -e|--env)
            ENV_NAME_ARG="$2"
            shift 2
            ;;
        -p|--prefix)
            ENV_PREFIX_ARG="$2"
            shift 2
            ;;
        --skip-sync)
            SKIP_SYNC=true
            shift
            ;;
        -o|--output)
            OUTPUT_ARG="$2"
            shift 2
            ;;
        -i|--inplace)
            INPLACE=true
            shift
            ;;
        -s|--session)
            SESSION_ARG="$2"
            shift 2
            ;;
        -l|--log)
            LOG_ARG="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -k|--kernel)
            KERNEL_ARG="$2"
            shift 2
            ;;
        -a|--attach)
            ATTACH=true
            shift
            ;;
        -*)
            echo "Error: Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            if [[ -z "$NOTEBOOK_ARG" ]]; then
                NOTEBOOK_ARG="$1"
                shift
            else
                echo "Error: Multiple notebook arguments provided: '$NOTEBOOK_ARG' and '$1'" >&2
                usage
                exit 1
            fi
            ;;
    esac
done

if ! command -v tmux >/dev/null 2>&1; then
    echo "Error: tmux is not installed or not in PATH." >&2
    exit 1
fi

PROJECT_DIR="$(pwd -P)"

PYPROJECT_NAME=""
if [[ -f "$PROJECT_DIR/pyproject.toml" ]]; then
    PYPROJECT_NAME="$(python3 -c '
import re
try:
    with open("pyproject.toml", "r") as f:
        match = re.search(r"name\s*=\s*[\"\x27]([^\"\x27]+)[\"\x27]", f.read())
        if match:
            print(match.group(1))
except Exception:
    pass
' 2>/dev/null || true)"
fi

TARGET_ENV_TYPE="venv"
TARGET_ENV_VAL="${PROJECT_DIR}/.venv"

if [[ -n "$ENV_NAME_ARG" ]]; then
    TARGET_ENV_TYPE="conda_name"
    TARGET_ENV_VAL="$ENV_NAME_ARG"
elif [[ -n "$ENV_PREFIX_ARG" ]]; then
    TARGET_ENV_TYPE="conda_prefix"
    TARGET_ENV_VAL="$(cd "$(dirname "$ENV_PREFIX_ARG")" && pwd -P)/$(basename "$ENV_PREFIX_ARG")"
elif [[ -n "$VENV_ARG" ]]; then
    TARGET_ENV_TYPE="venv"
    TARGET_ENV_VAL="$(cd "$(dirname "$VENV_ARG")" && pwd -P)/$(basename "$VENV_ARG")"
elif [[ -f "$PROJECT_DIR/.venv/bin/activate" ]]; then
    TARGET_ENV_TYPE="venv"
    TARGET_ENV_VAL="$PROJECT_DIR/.venv"
elif [[ -d "$PROJECT_DIR/.conda" && -d "$PROJECT_DIR/.conda/conda-meta" ]]; then
    TARGET_ENV_TYPE="conda_prefix"
    TARGET_ENV_VAL="$PROJECT_DIR/.conda"
elif [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "base" ]]; then
    TARGET_ENV_TYPE="conda_name"
    TARGET_ENV_VAL="$CONDA_DEFAULT_ENV"
else
    TARGET_ENV_TYPE="venv"
    TARGET_ENV_VAL="$PROJECT_DIR/.venv"
fi

KERNEL="${KERNEL_ARG:-python3}"

if [[ -z "$NOTEBOOK_ARG" ]]; then
    mapfile -t NB_FOUND < <(find . -maxdepth 3 -name "*.ipynb" -not -path "*/.*/*" | sort)
    if [[ ${#NB_FOUND[@]} -eq 1 ]]; then
        NOTEBOOK_ARG="${NB_FOUND[0]}"
        echo "Auto-detected notebook: $NOTEBOOK_ARG"
    elif [[ ${#NB_FOUND[@]} -gt 1 ]]; then
        echo "Error: No notebook specified. Available notebooks in current directory:" >&2
        for nb in "${NB_FOUND[@]}"; do
            echo "  $nb" >&2
        done
        usage
        exit 1
    else
        echo "Error: No notebook specified and no .ipynb files found in current directory." >&2
        usage
        exit 1
    fi
fi

if [[ ! -f "$NOTEBOOK_ARG" ]]; then
    echo "Error: Notebook file does not exist: $NOTEBOOK_ARG" >&2
    exit 1
fi

NOTEBOOK_ABS="$(cd "$(dirname "$NOTEBOOK_ARG")" && pwd -P)/$(basename "$NOTEBOOK_ARG")"
NB_DIR="$(dirname "$NOTEBOOK_ABS")"
NB_NAME="$(basename "$NOTEBOOK_ABS")"
NB_STEM="${NB_NAME%.ipynb}"
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"

if $INPLACE; then
    OUTPUT_ABS="$NOTEBOOK_ABS"
    EXEC_MODE="inplace"
elif [[ -n "$OUTPUT_ARG" ]]; then
    OUTPUT_DIR_PART="$(dirname "$OUTPUT_ARG")"
    mkdir -p "$OUTPUT_DIR_PART"
    OUTPUT_ABS="$(cd "$OUTPUT_DIR_PART" && pwd -P)/$(basename "$OUTPUT_ARG")"
    EXEC_MODE="output_file"
else
    OUTPUT_ABS="${NB_DIR}/${NB_STEM}.executed.ipynb"
    EXEC_MODE="output_file"
fi

if [[ -n "$LOG_ARG" ]]; then
    LOG_DIR_PART="$(dirname "$LOG_ARG")"
    mkdir -p "$LOG_DIR_PART"
    LOG_ABS="$(cd "$LOG_DIR_PART" && pwd -P)/$(basename "$LOG_ARG")"
else
    LOG_ABS="${NB_DIR}/${NB_STEM}_${TIMESTAMP}.run.log"
fi

if [[ -n "$SESSION_ARG" ]]; then
    SESSION_NAME="$SESSION_ARG"
else
    CLEAN_STEM="$(echo "$NB_STEM" | tr -c '[:alnum:]_' '_' | sed 's/__*/_/g' | sed 's/^_//;s/_$//')"
    SESSION_NAME="nb_${CLEAN_STEM}_${TIMESTAMP}"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Warning: tmux session '$SESSION_NAME' already exists. Appending process ID suffix."
    SESSION_NAME="${SESSION_NAME}_$$"
fi

RUNNER_SCRIPT="$(mktemp "${PROJECT_DIR}/.nb_run.XXXXXX.sh")"

cat > "$RUNNER_SCRIPT" << RUNNER_EOF
#!/usr/bin/env bash
set -u

PROJECT_DIR="${PROJECT_DIR}"
NOTEBOOK_ABS="${NOTEBOOK_ABS}"
OUTPUT_ABS="${OUTPUT_ABS}"
LOG_ABS="${LOG_ABS}"
EXEC_MODE="${EXEC_MODE}"
TIMEOUT="${TIMEOUT}"
KERNEL="${KERNEL}"
TARGET_ENV_TYPE="${TARGET_ENV_TYPE}"
TARGET_ENV_VAL="${TARGET_ENV_VAL}"
SKIP_SYNC="${SKIP_SYNC}"

trap 'rm -f "\$0"' EXIT

mkdir -p "\$(dirname "\$LOG_ABS")"
exec > >(tee -a "\$LOG_ABS") 2>&1

echo "Start Time:   \$(date -Iseconds)"
echo "Directory:    \$PROJECT_DIR"
echo "Notebook:     \$NOTEBOOK_ABS"
echo "Output:       \$OUTPUT_ABS"
echo "Log:          \$LOG_ABS"
echo "Kernel:       \$KERNEL"
echo "Timeout:      \$TIMEOUT seconds"
echo "Environment:  \$TARGET_ENV_TYPE (\$TARGET_ENV_VAL)"
echo "Runner PID:   \$\$"
echo ""

cd "\$PROJECT_DIR"

init_conda() {
    if [[ -n "\${CONDA_EXE:-}" && -x "\$CONDA_EXE" ]]; then
        eval "\$("\$CONDA_EXE" shell.bash hook 2>/dev/null)" && return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        eval "\$(conda shell.bash hook 2>/dev/null)" && return 0
    fi
    for candidate in \
        "\$HOME/miniconda3/bin/conda" \
        "\$HOME/.miniconda3/bin/conda" \
        "\$HOME/anaconda3/bin/conda" \
        "\$HOME/.anaconda3/bin/conda" \
        "/opt/conda/bin/conda" \
        "/opt/miniconda3/bin/conda" \
        "/opt/watchdog/users/flame/.miniconda3/bin/conda" \
        "/opt/watchdog/users/flame/miniconda3/bin/conda"; do
        if [[ -x "\$candidate" ]]; then
            export CONDA_EXE="\$candidate"
            eval "\$("\$candidate" shell.bash hook 2>/dev/null)" && return 0
        fi
    done
    return 1
}

if [[ "\$TARGET_ENV_TYPE" == "conda_name" || "\$TARGET_ENV_TYPE" == "conda_prefix" ]]; then
    if ! init_conda; then
        echo "Warning: Conda requested but not found. Falling back to .venv."
        TARGET_ENV_TYPE="venv"
        TARGET_ENV_VAL="\$PROJECT_DIR/.venv"
    fi
fi

if [[ "\$TARGET_ENV_TYPE" == "conda_name" ]]; then
    ENV_EXISTS=false
    if conda env list | awk '{print \$1}' | grep -Fxq "\$TARGET_ENV_VAL"; then
        ENV_EXISTS=true
    fi
    if ! \$ENV_EXISTS; then
        echo "Creating Conda environment '\$TARGET_ENV_VAL'..."
        conda create -y -n "\$TARGET_ENV_VAL" python=3.11 pip
    fi
    echo "Activating Conda environment: \$TARGET_ENV_VAL"
    conda activate "\$TARGET_ENV_VAL"
elif [[ "\$TARGET_ENV_TYPE" == "conda_prefix" ]]; then
    if [[ ! -d "\$TARGET_ENV_VAL/conda-meta" ]]; then
        echo "Creating Conda prefix at \$TARGET_ENV_VAL..."
        conda create -y -p "\$TARGET_ENV_VAL" python=3.11 pip
    fi
    echo "Activating Conda prefix: \$TARGET_ENV_VAL"
    conda activate "\$TARGET_ENV_VAL"
else
    if [[ ! -f "\$TARGET_ENV_VAL/bin/activate" ]]; then
        echo "Creating virtual environment at \$TARGET_ENV_VAL..."
        python3 -m venv "\$TARGET_ENV_VAL"
    fi
    echo "Activating virtual environment: \$TARGET_ENV_VAL"
    # shellcheck source=/dev/null
    source "\$TARGET_ENV_VAL/bin/activate"
fi

if [[ "\$SKIP_SYNC" != "true" && -f "\$PROJECT_DIR/pyproject.toml" ]]; then
    echo "Resolving dependencies from pyproject.toml..."
    DEPS_TMP="\$(mktemp "\$PROJECT_DIR/.pyproject_deps.XXXXXX.txt")"
    python - << 'PY_DEPS_EOF' > "\$DEPS_TMP"
import sys
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        try:
            import pip._vendor.tomli as tomllib
        except ImportError:
            tomllib = None

deps = []
if tomllib is not None:
    try:
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        deps.extend(data.get("project", {}).get("dependencies", []))
        deps.extend(data.get("dependency-groups", {}).get("dev", []))
    except Exception:
        pass

if not deps:
    import re
    try:
        with open("pyproject.toml", "r") as f:
            lines = f.readlines()
        in_deps = False
        for line in lines:
            line_s = line.strip()
            if line_s.startswith("dependencies = ["):
                in_deps = True
                continue
            if in_deps:
                if line_s.startswith("]"):
                    in_deps = False
                    continue
                m = re.search(r'["\x27]([^"\x27]+)["\x27]', line_s)
                if m:
                    deps.append(m.group(1))
    except Exception:
        pass

for dep in deps:
    print(dep)
PY_DEPS_EOF

    if [[ -s "\$DEPS_TMP" ]]; then
        echo "Installing pyproject.toml dependencies via pip..."
        python -m pip install --upgrade pip
        python -m pip install -r "\$DEPS_TMP"
    fi
    rm -f "\$DEPS_TMP"
fi

MISSING_TOOLS=()
python -c "import jupyter" >/dev/null 2>&1 || MISSING_TOOLS+=("jupyter")
python -c "import nbconvert" >/dev/null 2>&1 || MISSING_TOOLS+=("nbconvert")
python -c "import ipykernel" >/dev/null 2>&1 || MISSING_TOOLS+=("ipykernel")

if [[ \${#MISSING_TOOLS[@]} -gt 0 ]]; then
    echo "Installing missing Jupyter tools (\${MISSING_TOOLS[*]}) into environment..."
    python -m pip install "\${MISSING_TOOLS[@]}"
fi

python -m ipykernel install --user --name "\$KERNEL" --display-name "Python (\$KERNEL)" >/dev/null 2>&1 || true

START_EPOCH=\$(date +%s)
STATUS=0

echo ""
echo "Beginning headless notebook execution..."
if [[ "\$EXEC_MODE" == "inplace" ]]; then
    jupyter nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.timeout="\$TIMEOUT" \
        --ExecutePreprocessor.kernel_name="\$KERNEL" \
        "\$NOTEBOOK_ABS" || STATUS=\$?
else
    OUT_DIR="\$(dirname "\$OUTPUT_ABS")"
    OUT_FILE="\$(basename "\$OUTPUT_ABS")"
    mkdir -p "\$OUT_DIR"
    jupyter nbconvert \
        --to notebook \
        --execute \
        --output "\$OUT_FILE" \
        --output-dir "\$OUT_DIR" \
        --ExecutePreprocessor.timeout="\$TIMEOUT" \
        --ExecutePreprocessor.kernel_name="\$KERNEL" \
        "\$NOTEBOOK_ABS" || STATUS=\$?
fi

END_EPOCH=\$(date +%s)
DURATION=\$((END_EPOCH - START_EPOCH))

echo ""
if [[ \$STATUS -eq 0 ]]; then
    echo "Execution completed successfully in \${DURATION}s"
    echo "Output notebook saved: \$OUTPUT_ABS"
else
    echo "Execution failed with exit code \$STATUS after \${DURATION}s"
fi
echo "Finish Time: \$(date -Iseconds)"

echo ""
echo "Session finished. Press Enter or exit to terminate tmux window."
read -r _ || true
exit \$STATUS
RUNNER_EOF

chmod +x "$RUNNER_SCRIPT"

tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_DIR" "$RUNNER_SCRIPT"

echo "Notebook execution started in tmux session: $SESSION_NAME"
echo ""
echo "Summary:"
echo "  Notebook:     $NOTEBOOK_ABS"
echo "  Output:       $OUTPUT_ABS"
echo "  Log File:     $LOG_ABS"
echo "  Environment:  $TARGET_ENV_TYPE ($TARGET_ENV_VAL)"
echo "  Tmux Session: $SESSION_NAME"
echo ""
echo "Monitoring & Management:"
echo "  Attach session:   tmux attach -t $SESSION_NAME"
echo "  Tail live logs:   tail -f $LOG_ABS"
echo "  List sessions:    tmux ls"
echo "  Kill session:     tmux kill-session -t $SESSION_NAME"

if $ATTACH; then
    echo ""
    echo "Attaching to session: $SESSION_NAME..."
    tmux attach -t "$SESSION_NAME"
fi
