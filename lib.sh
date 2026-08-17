# Shared helpers for daily.sh and weekly.sh.
# Source with: source "$(dirname "$0")/lib.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
SITE_BASE_URL="https://sampo.website/ainikki"

validate_env() {
    if [ -z "${OPENROUTER_API_KEY:-}" ]; then
        echo "Error: OPENROUTER_API_KEY is not set." >&2
        exit 1
    fi
    if [ -z "${DEPLOY_TARGET:-}" ]; then
        echo "Error: DEPLOY_TARGET is not set." >&2
        exit 1
    fi
}

send_error() {
    local msg="$1"
    if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_ERROR_CHAT_ID:-}" ]; then
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            --data-urlencode "text=${msg}" \
            -d chat_id="${TELEGRAM_ERROR_CHAT_ID}" > /dev/null
    fi
}

run_pipeline() {
    local topic="$1" period="$2" since="$3" until="$4" display_date="$5"
    local log pipe_exit

    log=$(mktemp)
    pipe_exit=0
    "$VENV_PYTHON" -m agent.pipeline \
        --topic "$topic" --period "$period" \
        --since "$since" --until "$until" \
        --display-date "$display_date" --verbose \
        2> >(tee "$log" >&2) || pipe_exit=$?

    if [ "$pipe_exit" -ne 0 ]; then
        local err_msg
        err_msg=$(tail -1 "$log")
        rm -f "$log"
        send_error "Ainikki epäonnistui ($topic $since): $err_msg"
        exit "$pipe_exit"
    fi
    rm -f "$log"
}

deploy_site() {
    local log deploy_exit
    log=$(mktemp)
    deploy_exit=0
    "$SCRIPT_DIR/site/deploy.sh" 2> >(tee "$log" >&2) || deploy_exit=$?

    if [ "$deploy_exit" -ne 0 ]; then
        local err_msg
        err_msg=$(tail -1 "$log")
        rm -f "$log"
        send_error "Ainikki deploy epäonnistui: $err_msg"
        exit "$deploy_exit"
    fi
    rm -f "$log"
}

send_telegram() {
    local topic="$1" period="$2" display_date="$3" chat_id="$4"
    local json_file link overview digest_topic msg

    json_file="data/output/${topic}_${period}_${display_date}.json"
    if [ ! -f "$json_file" ]; then
        echo "Warning: ${json_file} not found, Telegram message not sent." >&2
        return
    fi

    overview=$("$VENV_PYTHON" -c "
import json, sys
d = json.load(open('$json_file'))
print(d.get('overview', 'No overview available.'))
")
    digest_topic=$("$VENV_PYTHON" -c "
import json, sys
d = json.load(open('$json_file'))
print(d.get('digest_topic', ''))
")
    link="${SITE_BASE_URL}/${topic}/${display_date}/"
    nl=$'\n'
    if [ -n "$digest_topic" ]; then
        msg="${digest_topic}${nl}${nl}${overview}${nl}${nl}${link}"
    else
        msg="${overview}${nl}${nl}${link}"
    fi

    echo "== Sending Telegram notification =="
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d chat_id="${chat_id}" \
        --data-urlencode "text=${msg}" > /dev/null
    echo "Telegram: done."
}