# Shared helpers for daily.sh and weekly.sh.
# Source with: source "$(dirname "$0")/lib.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
SITE_BASE_URL="${SITE_BASE_URL:-https://sampo.website/ainikki}"

# Muunnetaan MP3 OGG Opusiksi (Telegram sendVoice vaatii OGG).
# # TTS tuottaa NyKYääN OGG:n suoraan; tämä toimii vain vanhojen .mp3-tiedostojen jälkikäsittelyyn.
# Jos ffmpeg ei löydy, palautetaan alkuperäinen polku.
prepare_audio() {
    local path="$1"
    if [[ "$path" == *.ogg ]]; then
        echo "$path"
        return
    fi
    if [[ "$path" == *.mp3 ]] && command -v ffmpeg &>/dev/null; then
        local ogg_path="${path%.mp3}.ogg"
        ffmpeg -y -i "$path" -c:a libopus -b:a 64k "$ogg_path" >/dev/null 2>&1
        if [[ -s "$ogg_path" ]]; then
            path="$ogg_path"
        fi
    fi
    echo "$path"
}

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
    local topic="$1" period="$2" display_date="$3" chat_id="$4" audio_path="$5"

    local json_file="data/output/${topic}_${period}_${display_date}.json"
    if [ ! -f "$json_file" ]; then
        echo "Warning: ${json_file} not found, Telegram message not sent." >&2
        return
    fi

    local item_count link overview digest_topic msg nl
    nl=$'\n'

    item_count=$(jq '.items | length' "$json_file")

    overview=$(jq -r '.overview // "No overview available."' "$json_file")
    digest_topic=$(jq -r '.digest_topic // ""' "$json_file")

    msg="$overview"
    if [ -n "$item_count" ] && [ "$item_count" -gt 0 ] 2>/dev/null; then
        msg="${msg}${nl}${nl}Lue koko kooste (${item_count} uutista):"
    else
        msg="${msg}${nl}${nl}Lue koko kooste:"
    fi

    if [ -n "$digest_topic" ]; then
        msg="${nl}${nl}${nl}${digest_topic}${nl}${nl}${msg}"
    fi

    if [ -n "$SITE_BASE_URL" ]; then
        link="${SITE_BASE_URL}/${topic}/${display_date}/"
        msg="${msg}${nl}${link}"
    fi

    echo "== Sending Telegram notification =="
    if [ -n "$audio_path" ] && [ -f "$audio_path" ]; then
        response=$(curl -s --max-time 30 -w "\nHTTP_CODE:%{http_code}" -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendVoice" \
            -F "voice=@${audio_path}" \
            -F "caption=${msg}" \
            -F "chat_id=${chat_id}" 2>&1)
    else
        response=$(curl -s --max-time 30 -w "\nHTTP_CODE:%{http_code}" -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -F "chat_id=${chat_id}" \
            -F "text=${msg}")
    fi
    echo "Telegram: done."
}
