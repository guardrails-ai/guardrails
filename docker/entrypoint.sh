#!/bin/sh
set -e

# Runtime extension installation (like Grafana's GF_INSTALL_PLUGINS pattern)
# Usage: GUARDRAILS_EXTENSIONS="guardrails/two_words,guardrails/other" GUARDRAILS_TOKEN="..."
if [ -n "${GUARDRAILS_EXTENSIONS}" ]; then
    if [ -z "${GUARDRAILS_TOKEN}" ]; then
        echo "ERROR: GUARDRAILS_EXTENSIONS is set but GUARDRAILS_TOKEN is not provided" >&2
        exit 1
    fi

    echo "Configuring Guardrails Hub authentication..."
    guardrails configure --enable-metrics --enable-remote-inferencing --token "${GUARDRAILS_TOKEN}"

    echo "Installing extensions: ${GUARDRAILS_EXTENSIONS}"
    echo "${GUARDRAILS_EXTENSIONS}" | tr ',' '\n' | while IFS= read -r extension; do
        if [ -n "${extension}" ]; then
            # Add hub:// prefix if not present
            case "${extension}" in
                hub://*) uri="${extension}" ;;
                *) uri="hub://${extension}" ;;
            esac
            echo "  -> Installing ${uri}..."
            guardrails hub install "${uri}"
        fi
    done

    echo "Extension installation complete."
fi

# Apply guard template if specified and not already applied
if [ -n "${GUARDRAILS_TEMPLATE}" ] && [ -f "/app/${GUARDRAILS_TEMPLATE}" ]; then
    echo "Applying guard template: ${GUARDRAILS_TEMPLATE}"
    # Use yes to auto-confirm any prompts
    yes | guardrails create --template "/app/${GUARDRAILS_TEMPLATE}" 2>/dev/null || true
fi

# Execute the main command (uvicorn)
exec "$@"
