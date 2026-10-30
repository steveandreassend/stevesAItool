# Steve's AI Tool

A local-first research and analysis application for Apple Silicon Macs. It combines local MLX language models, Exa web search, document analysis, multimedia audio analysis, image and video analysis, webpage analysis, YouTube summarization, live voice input, multilingual read-aloud, and a browser-based interface in a single Python application.

The application is designed for research workflows where you want local model inference and direct control over the evidence supplied to the model, while retaining the option to retrieve current web content through Exa.

<p align="center">
  <img src="/images/UI.png" width="800" alt=Overview of the user interface.">
  <br>
  <em>Overview of the user interface.</em>
</p>

## Features

**Hardware:**
- **Private** - It runs entirely on a MacBook Pro and no AI prompts or reasoning information leaves it. And it’s free, without user limits.
- **Apple optimized** - Exploits the Apple silicon hardware capabilities to improve performance (MLX).
- **Monitoring** - Monitors the memory consumption of your laptop in case you overload it.

**Search:**
- **Controlled search** - Personalized search, with user controls over which content to reason upon to control accuracy, bias, and the location where you’re searching from.
- **Suggested searches** - Generates suggested additional searches to retrieve more related content, based upon what your stated objective is.
- **Combine searches** - Allows you to run multiple searches and combine the results.
- **Control sources** - Allows you to include and exclude specific content and websites. Allows you to remove specific search results from the reasoning phase.
- **Quantity of sources** - Control how many articles you want to retrieve. Commercial services might retrieve 2-3, but what if you want 80 to capture a broader view on a topic?
- **Previews** - Toggle whether you want to see a preview of the sources.

**Languages:**
- **Multilingual** - Reads, writes, speaks, and listens in multiple languages in a native dialect and accent.
- **Voice control** - Live AI chat mode, with a mic mute button.

**Reasoning:**
- **Bulk document analysis** - Upload up to 15 documents at a time. Copilot limits you to two.
- **Multimedia audio analysis** - Extract the audio from any video or audio file, and reason upon it.
- **Image and video analysis** - Examine any image or view file, and reason upon it with two different models.
- **Webpage analysis** - Specify the URL to any article in any language, and reason upon it.
- **Model selection** - Choice of models from Google, Qwen, Deepseek depending on the chosen task - whether it be fast summarization or slow deep reasoning.
- **Switch models** - Switch between models during the chat.
- **Ease of use** - Automatically downloads a model from Hugging Face the first time you select it.
- **Large context windows** - Up to an 8X larger context window (262K) than some of the major commercial services (16K).
- **Youtube transcription** - Transcribes the audio of any Youtube video in any language, even if there are no subtitles, for analysis and reasoning. No more sitting through long seminars or lectures. Control which segment of the audio to analyze by defining the start and end boundaries.
- **Reasoning** - Shows the reasoning so you can understand how conclusions were reached.
- **Notifications** - Notifications - it rings a bell when it’s done. You can mute that too.

**Downsides:**
- **Speed** - Responses take considerably longer than using a powerful GPU cluster in the cloud. Usually you'll have time to get a coffee for each response.
- **Investment** - You’ll need a MacBook that costs in excess of 3000 EUR as a starting point, with at least 24GB RAM as an absolute minimum. Budget 5000 EUR for a good one. Shop around - some Windows laptops with Nvidia or AMD chips can offer much better performance than a Mac and may cost you less.

## Table of contents

- [Core capabilities](#core-capabilities)
  - [Local model inference](#local-model-inference)
  - [Exa web research](#exa-web-research)
  - [Local document analysis](#local-document-analysis)
  - [YouTube summarization](#youtube-summarization)
  - [Live voice mode](#live-voice-mode)
  - [Multilingual read-aloud](#multilingual-read-aloud)
  - [Research-oriented browser interface](#research-oriented-browser-interface)
- [Architecture](#architecture)
- [Requirements](#requirements)
  - [Hardware](#hardware)
  - [Software](#software)
- [Installation on macOS](#installation-on-macos)
- [Recommended first run](#recommended-first-run)
- [Configuration](#configuration)
- [Model keys](#model-keys)
- [Using the application](#using-the-application)
- [Privacy and data handling](#privacy-and-data-handling)
- [Important limitations](#important-limitations)
- [Troubleshooting](#troubleshooting)
- [Security recommendations before remote exposure](#security-recommendations-before-remote-exposure)
- [Licence](#licence)
- [Acknowledgements](#acknowledgements)


## Usage Screenshots

<p align="center">
  <img src="/images/Search1.png" width="800" alt="Searching content for reasoning using Exa, specifying a search and an objective.">
  <br>
  <em>Searching content for reasoning using Exa, specifying a search and an objective.</em>
</p>

<br>

<p align="center">
  <img src="/images/Search2.png" width="800" alt="De-select the search results to exclude from reasoning">
  <br>
  <em>De-select the search results to exclude from reasoning</em>
</p>

<br>

<p align="center">
  <img src="/images/Search3.png" width="800" alt="Receive suggested additional searches based upon the previous search results, taking into account the original search objective.">
  <br>
  <em>Receive suggested additional searches based upon the previous search results, taking into account the original search objective.</em>
</p>

<br>

<p align="center">
  <img src="/images/Search4.png" width="800" alt="Example of the reasoning output, model choice, live chat, and text to speech">
  <br>
  <em>Example of the reasoning output, model choice, live chat, and text to speech</em>
</p>

<br>

<p align="center">
  <img src="/images/Search5.png" width="800" alt="Citations of the selected articles used for reasoning, and the reasoning time.">
  <br>
  <em>Citations of the selected articles used for reasoning, and the reasoning time.</em>
</p>

## Core capabilities

### Local model inference

Choose a model for each prompt:

- **Qwen3 14B 4-bit (DEFAULT)** — general-purpose research, analysis, synthesis, writing, and multilingual tasks. Supports **32,768 tokens natively**, extendable to **131,072 tokens with YaRN**, with switchable thinking and non-thinking modes.
- **DeepSeek-R1 Distill Qwen 14B 4-bit** — optimized for deliberate reasoning, complex problem-solving, mathematics, coding, and tasks that benefit from self-verification. Supports approximately **128K tokens**, but is generally slower and more verbose than Qwen3.
- **Qwen3.5 9B 4-bit** — suited to long-document summarization, research synthesis, visual-document analysis, and tasks requiring a strong balance of quality, speed, and memory use. Supports **262,144 tokens natively**, extendable to approximately **1 million tokens**, with text and image input.
- **Qwen3.5 4B 4-bit** — the fastest and most memory-efficient option for summarization, extraction, classification, and routine document processing. Supports **262,144 tokens natively**, extendable to approximately **1 million tokens**, with text and image input. In practice, the usable context length depends on available unified memory.
- **Qwen3-VL 8B Instruct 4-bit** — the preferred model for screenshots, OCR, financial tables, charts, scanned documents, visual interfaces, and other image-intensive analysis. Supports a native context window of **256K tokens**, extendable to approximately **1 million tokens**, with strong visual grounding, spatial understanding, and multilingual OCR.
- **Gemma 4 12B 4-bit** — designed for broader multimodal analysis and reasoning across text, images, audio, and video through **MLX-VLM**. Supports a context window of up to **256K tokens** and configurable thinking modes. Use it for multimodal reasoning that extends beyond screenshots and visual-document extraction.

Only one large model is retained in memory at a time. Model selection applies to the current prompt only. When switching models, the application releases the previous model and clears the MLX cache before loading the next one. The selected model receives up to the 12 most recent user-assistant exchanges from the shared session history. For standard chat prompts, it also receives the currently selected search results and attached files or images. Direct article, YouTube, and uploaded-media analyses use task-specific contexts instead, although their completed outputs are added to the shared conversation history for subsequent standard prompts.

**Why this is useful:**

- Prompts and uploaded document text remain on the Mac during inference.
- Different models can be selected according to speed, context size, and reasoning needs.
- Single-model memory management reduces the risk of exhausting unified memory when switching between models.
- Reasoning output is separated from the final response and displayed in a collapsible section when available.
- Ability to feed previous responses and search results processed with other models to a new model for enhancing and refining the analysis.

### Exa web research

The integrated Exa search interface supports:

- Up to 80 results per search, in increments of eight.
- Multiple searches in the same research session.
- Cumulative results across search batches.
- URL deduplication between searches.
- Per-result inclusion or exclusion before model analysis.
- Search objectives to improve ranking and retrieval focus.
- Auto, Instant, Fast, Deep Lite, Deep, and Deep Reasoning search modes.
- Country-aware search using a two-letter location code.
- Search categories including news, publications, companies, people, financial reports, and personal sites.
- Website inclusion and exclusion filters.
- Cached search, live trawl, publication-age, and publication-date-range controls.
- Search across one or more selected languages, with English selected by default.
- Optional result highlights.
- Model-generated suggestions for three complementary follow-up searches.
- Search time, result totals, duplicate counts, and Exa cost reporting.

**Why this is useful:**

- Research can be built in stages instead of replacing earlier results with every search.
- You decide exactly which sources are passed to the model.
- The model analyzes the selected evidence rather than treating retrieval output as the final answer.
- Language, domain, category, location, and timeframe controls make searches more targeted.
- Follow-up suggestions help expose missing evidence, weak coverage, or contradictory sources.

> Exa is the only core feature that requires a third-party API key and an internet connection. Local model inference does not require an external model API after the model files have been downloaded.

### Webpage analysis

Paste the article to any URL and the selected LLM will examine it and respond to your prompt.

**Why this is useful:**

- Analyze and reason upon webpages, which is a feature that commercial tools might refuse to offer.

### Local multimedia analysis

Attach a image or video file and reason upon it with Gemma or Qwen 3 VL.

**Why this is useful:**

- Analyze and reason upon multimedia.

### Local multimedia analysis

Attach a media file and the audio will be transcribed and examined by the LLM along with the prompt. Supported formats are any types supported by ffmpeg.

### Local document analysis

Attach up to 15 documents in one upload. Supported formats are:

- `.txt`
- `.md`
- `.csv`
- `.json`
- `.py`
- `.log`
- `.html`
- `.xml`
- `.pdf`
- `.docx`
- `.xlsx`

Text is extracted locally and added to the active research session. PDF, Word, and Excel extraction are handled locally with Python libraries.

**Why this is useful:**

- Internal documents can be analyzed alongside selected web evidence.
- Multiple files can be compared in one prompt.
- The application does not require documents to be sent to a hosted LLM provider.

> The default extraction limit is 120,000 characters per file. Scanned PDFs require OCR before upload because the application extracts embedded PDF text but does not perform OCR.

### YouTube summarization

Paste a supported single-video YouTube URL into the prompt field. The user is prompted for the start and end of the audio segment to analyze, defaulting to the entire clip. The application:

1. Attempts to retrieve an available transcript.
2. Falls back to downloading the audio and transcribing it locally with MLX Whisper.
3. Produces an executive summary, detailed key points, and conclusions.
4. Stores transcript and summary text temporarily under `temp/youtube/`.
5. Deletes downloaded media after processing.

**Why this is useful:**

- Long videos can be converted into decision-useful briefings.
- Local Whisper transcription provides a fallback when captions are unavailable.
- The summary is generated by the selected local model.

### Live voice mode

Live mode captures speech from the Mac microphone, detects the end of an utterance, transcribes it locally with MLX Whisper, and submits the prompt to the selected model.

It includes:

- Adaptive voice activity detection.
- Configurable silence and maximum-utterance thresholds.
- Manual microphone mute and unmute.
- Automatic microphone muting before read-aloud playback.
- Live transcript and processing status in the browser.
- Optional completion sound.

**Why this is useful:**

- It enables hands-free querying and research.
- Transcription runs locally.
- Adaptive detection is more practical than requiring a fixed push-to-talk duration.

### Multilingual read-aloud

Responses can be read aloud in:

- English
- French
- Spanish
- German
- Italian
- Dutch
- Portuguese

English uses Piper with NeMo Text Processing for text normalization. Non-English playback uses offline Argos Translate for English-to-target translation and a language-specific Piper voice. Piper voice models can be downloaded automatically on first use.

**Why this is useful:**

- Research output can be consumed without remaining at the screen.
- Translation and non-English playback can operate locally after language packages and voices are installed.
- Playback speed is adjustable from 0.75x to 2.0x.

### Research-oriented browser interface

The Flask application serves a responsive local web interface with:

- Light and dark themes.
- Persistent theme selection.
- Markdown rendering for model responses.
- Formatted response copying.
- Collapsible model reasoning.
- Source links displayed separately from the answer.
- Sticky memory-health reporting updated every 10 seconds.
- Search-result selection and cumulative batch display.
- Session reset and temporary-file cleanup.

## Architecture

The project is intentionally contained in a single `app.py` file:

- **Flask** provides the local server and JSON endpoints.
- **Embedded HTML, CSS, and JavaScript** provide the browser interface.
- **MLX-LM and MLX-VLM** run local language models.
- **Exa** supplies optional web retrieval.
- **MLX Whisper** handles voice and fallback YouTube transcription.
- **Piper, NeMo Text Processing, and Argos Translate** provide local speech, text-normalization, and translation functions.
- **Python document libraries** extract text from common file formats.

This single-file structure makes the application easy to inspect and deploy locally, although production deployments may benefit from separating the server, templates, static assets, configuration, and dependency definitions.

## Requirements

### Hardware

- Apple Silicon Mac: M-series processor recommended.
- At least 24 GB of unified memory for the smaller models.
- 32 GB or more recommended for comfortable use of 12B to 14B 4-bit models.
- Additional free disk space for model weights, Whisper, speech models, and caches.

Actual memory use and performance depend on the selected model, prompt length, retrieved evidence, and other applications running on the Mac.

### Software

- macOS
- Python 3.11 recommended
- Homebrew
- Git
- FFmpeg
- PortAudio
- An Exa API key for web search

## Installation on macOS

### 1. Install system dependencies

Install Homebrew and Miniconda if they are not already available, then install the required command-line dependencies:

```bash
brew update
brew install git ffmpeg portaudio
brew install --cask miniconda
```

`ffmpeg` is required for YouTube audio conversion. `portaudio` is required by the live microphone feature through `sounddevice`.

Initialize Conda for `zsh` if required, then restart Terminal or reload the shell configuration:

```bash
conda init zsh
source ~/.zshrc
```

Confirm the tools are available:

```bash
conda --version
git --version
ffmpeg -version
```

### 2. Clone the repository

Replace the URL below with the URL of your GitHub repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

### 3. Create and activate the Conda environment

Create a dedicated environment and activate it:

```bash
conda create -n local-assistant python=3.13.5 -y
conda activate local-assistant
```

Upgrade the packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Whenever you open a new Terminal window, return to the repository and reactivate the environment:

```bash
conda activate local-assistant
```

### 4. Install Python dependencies

Install Pynini through Conda. The tested version is `2.1.6.post1`:

```bash
conda install -c conda-forge "pynini=2.1.6.post1" -y
```

Install the main application, model, document, voice, translation, and YouTube dependencies:

```bash
python -m pip install \
  flask \
  werkzeug \
  mlx \
  mlx-lm \
  mlx-vlm \
  mlx-whisper \
  piper-tts \
  yt-dlp \
  youtube-transcript-api \
  argostranslate \
  pypdf \
  python-docx \
  openpyxl \
  sounddevice \
  pydub \
  psutil \
  numpy \
  nltk \
  trafilatura \
  beautifulsoup4 \
  lxml \
  Pillow
```

Install NeMo Text Processing without dependency resolution, then install the additional libraries it requires:

```bash
python -m pip install "nemo_text_processing==1.2.0" --no-deps

python -m pip install \
  inflect \
  editdistance \
  sacremoses \
  pandas \
  regex \
  cdifflib \
  transformers \
  wget \
  wrapt
```

Check the environment for unresolved or incompatible dependencies:

```bash
python -m pip check
```

Verify Pynini and NeMo Text Processing, including the English normalization used before speech synthesis:

```bash
python - <<'PY'
import pynini
import cdifflib
import transformers
import wget
import wrapt

from nemo_text_processing.text_normalization.normalize import Normalizer

print("Pynini:", pynini.__version__)
normalizer = Normalizer(input_case="cased", lang="en")

for text in ["AI", "WW II", "AI played an important role in WW II."]:
    print(text, "->", normalizer.normalize(text, verbose=False))
PY
```

If Pynini was previously installed at another version, replace it with the tested build:

```bash
conda remove pynini -y
conda install -c conda-forge "pynini=2.1.6.post1" -y
```

### 5. Configure the Exa API key

Set the key for the current Terminal session:

```bash
export EXA_API_KEY="your-exa-api-key"
```

To load it automatically in future shells, add it to `~/.zshrc`:

```bash
printf '\nexport EXA_API_KEY="your-exa-api-key"\n' >> ~/.zshrc
source ~/.zshrc
```

Do not commit an API key to GitHub.

A safer repository workflow is to use an untracked environment file:

```bash
cat > .env <<'EOF'
EXA_API_KEY=your-exa-api-key
EOF
```

The current application does not automatically load `.env`, so either export the value before launch or add a dotenv loader. Ensure `.env` is listed in `.gitignore`.

### 6. Start the application

```bash
python app.py
```

Open the local interface in a browser:

```text
http://127.0.0.1:8080
```

The first use of a model can take time because its files must be downloaded. Subsequent launches normally reuse the local model cache.

### 7. Allow microphone access

When using live mode for the first time, macOS may request microphone permission for Terminal, iTerm, or the Python host process.

If access was denied:

1. Open **System Settings**.
2. Go to **Privacy & Security > Microphone**.
3. Enable access for the terminal application used to start the server.
4. Restart the application.

## Recommended first run

1. Start with **Quick Summarization — Qwen3.5 4B** to verify model loading with lower memory pressure.
2. Enter a short prompt and select **Run prompt**.
3. Attach a small text or PDF file and ask for a structured summary.
4. Add `EXA_API_KEY`, run an eight-result English search, and inspect the retrieved highlights.
5. Deselect one or more results and ask the model to analyze only the selected evidence.
6. Test larger models after confirming that the basic workflow is stable.

## Configuration

The application supports the following environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `EXA_API_KEY` | none | Enables Exa web search. |
| `DEFAULT_MODEL_KEY` | `default` | Initial model key: `default`, `reasoning`, `summary`, `quick`, or `gemma`. |
| `HOST` | `127.0.0.1` | Flask bind address. Keep this local unless you implement authentication and hardening. |
| `PORT` | `8080` | Local web-server port. |
| `WHISPER_MODEL` | `mlx-community/whisper-large-v3-turbo` | MLX Whisper model used for transcription. |
| `MAX_FILE_CHARS` | `120000` | Maximum extracted characters retained per uploaded file. |
| `MAX_HISTORY_TURNS` | `12` | Number of recent dialogue turns retained in the session. |
| `MAX_WEB_RESULTS` | `8` | Default maximum Exa results. The UI requests results in multiples of eight. |
| `EXA_QUERY_HISTORY_TURNS` | `6` | Recent turns available to contextual search-query planning. |
| `EXA_QUERY_MAX_TOKENS` | `180` | Token limit for generated search queries. |
| `MAX_TRANSCRIPT_CHARS` | `500000` | Maximum transcript characters supplied for YouTube summarization. |
| `LIVE_MIN_THRESHOLD` | `0.006` | Minimum live-mode audio threshold. |
| `LIVE_SILENCE_SECONDS` | `1.75` | Silence duration used to end an utterance. |
| `LIVE_MAX_UTTERANCE_SECONDS` | `30` | Maximum duration of one live utterance. |
| `ARGOS_AUTO_DOWNLOAD` | `1` | Automatically install missing Argos language packages. |
| `PIPER_AUTO_DOWNLOAD` | `1` | Automatically download missing Piper voices. |
| `PIPER_USE_CUDA` | `0` | Piper CUDA option. Leave disabled on a Mac. |
| `PIPER_LENGTH_SCALE` | `1.0` | Piper speech-length scale. |
| `PIPER_SENTENCE_SILENCE` | `0.25` | Piper pause between sentences. |

Piper voice names can also be overridden with:

- `PIPER_VOICE_FR`
- `PIPER_VOICE_DE`
- `PIPER_VOICE_ES`
- `PIPER_VOICE_IT`
- `PIPER_VOICE_NL`
- `PIPER_VOICE_PT`

Example:

```bash
export DEFAULT_MODEL_KEY="quick"
export PORT="8081"
export MAX_FILE_CHARS="200000"
python app.py
```

## Model keys

| Key | Model | Backend | Typical role |
|---|---|---|---|
| `default` | `mlx-community/Qwen3-14B-4bit` | MLX-LM | General research, analysis, writing, and synthesis |
| `reasoning` | `mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit` | MLX-LM | Deliberate reasoning and complex problem-solving |
| `summary` | `mlx-community/Qwen3.5-9B-MLX-4bit` | MLX-VLM | Long-context summarization and visual-document analysis |
| `quick` | `mlx-community/Qwen3.5-4B-MLX-4bit` | MLX-VLM | Faster, lower-memory summarization and extraction |
| `gemma` | `mlx-community/gemma-4-12B-it-4bit` | MLX-VLM | Broad multimodal analysis across text, images, audio, and video |
| `qwen3_vl` | `mlx-community/Qwen3-VL-8B-Instruct-4bit` | MLX-VLM | Screenshots, OCR, financial tables, charts, and visual-document analysis |

Model identifiers, repositories, and compatibility can change. If a configured model is unavailable, update the corresponding entry in `MODEL_OPTIONS` in `app.py` to a compatible MLX repository.

## Using the application

### A. Search first, then analyze

1. Enter a search query.
2. Optionally define a search objective.
3. Set language, location, category, domain, freshness, and result limits.
4. Select **Search**.
5. Review the results and highlights.
6. Deselect sources that should not be used.
7. Enter an analytical prompt, choose a model, and select **Run prompt**.

All selected results from all accumulated search batches are available to the model.

### B. Build a multi-search evidence set

Run several related searches. Each new batch is appended to the existing session, while duplicate URLs are skipped. After each batch, use **Suggest follow up searches** to identify complementary research angles. Select the articles that should form the final evidence set, then ask the model to compare or synthesize them.

### C. Analyze local documents

Select **Attach up to 15 documents**, choose the files, and wait for extraction to complete. Then ask the model to summarize, compare, challenge, or combine the uploaded material. Files remain attached until the session is reset.

### D. Summarise a YouTube video

Paste a supported YouTube video URL as the complete prompt and run it. If captions are unavailable, the application attempts local audio transcription. Processing time depends on video length and Mac performance.

### E. Use live voice mode

Choose the desired model and select **Start live mode**. Speak normally and pause when finished. The application transcribes the utterance and produces a response. Use **Mute mic** when required, and stop live mode when finished.

### F. Read a response aloud

Choose a playback language beside a response and select **Read aloud**. For non-English languages, the first request may take longer while Argos packages and Piper voices are downloaded.

## Privacy and data handling

The local-first design reduces unnecessary disclosure, but it should not be treated as a complete security boundary.

- Model inference is local after model downloads.
- Uploaded document extraction is local.
- Voice transcription is local.
- Exa search queries, search settings, and retrieval requests are sent to Exa when search is used.
- YouTube transcript retrieval or media downloading communicates with YouTube and related services.
- Model, voice, translation, and Python package downloads contact their respective repositories.
- Session state is kept in application memory.
- Uploaded files and generated audio are stored below `temp/` during use.
- Reset removes the active session's uploaded-file directory and generated audio.
- The server has no authentication and binds to localhost by default.

For confidential or regulated use, review the code, dependency supply chain, data-retention behavior, Exa terms, and network traffic before deployment.

## Important limitations

- This is a local development application intended for private use, not a hardened multi-user service.
- In-memory sessions are lost when the process stops.
- There is no user authentication or authorization.
- Uploaded filenames are sanitized, but uploaded content should still be treated as untrusted input.
- Prompt injection can exist inside web pages and uploaded documents. Review selected evidence and do not assume model output is authoritative.
- Extracted PDF text quality depends on the source PDF.
- Excel extraction reads cell values but does not reproduce workbook formatting or formulas as a visual spreadsheet.
- YouTube access and transcript availability can change.
- Inference speed depends heavily on the Mac model, unified memory, model size, and context length.
- Exa searches may incur usage charges.
- The application displays source links, but final claims should still be checked against the original material.

## Troubleshooting

### `EXA_API_KEY is not set`

Export the key in the same shell before starting the application:

```bash
export EXA_API_KEY="your-exa-api-key"
python app.py
```

### MLX or Metal out-of-memory error

- Close memory-intensive applications.
- Switch to the 4B model.
- Reduce the number of selected search results.
- Use shorter prompts and fewer or smaller documents.
- Restart the Python process to fully release memory.
- Monitor the memory status displayed at the top of the interface.

### `mlx-vlm` is required

```bash
python -m pip install --upgrade mlx-vlm
```

### FFmpeg or YouTube audio conversion failure

```bash
brew install ffmpeg
ffmpeg -version
```

Then update `yt-dlp`:

```bash
python -m pip install --upgrade yt-dlp
```

### Microphone does not work

Install or relink PortAudio:

```bash
brew install portaudio
python -m pip install --upgrade sounddevice
```

Then check **System Settings > Privacy & Security > Microphone** and allow access for the terminal application.

### NeMo Text Processing is unavailable

Confirm that Pynini is installed through Conda, then reinstall NeMo Text Processing and its required libraries:

```bash
conda install -c conda-forge "pynini=2.1.6.post1" -y
python -m pip install "nemo_text_processing==1.2.0" --no-deps
python -m pip install inflect editdistance sacremoses pandas regex cdifflib transformers wget wrapt
python -m pip check
```

### Piper voice download fails

Download a configured voice manually. For example:

```bash
python -m piper.download_voices \
  --download-dir voices/piper \
  fr_FR-siwis-medium
```

### Argos translation package is unavailable

By default, the application attempts to download missing English-to-target packages. Confirm that `ARGOS_AUTO_DOWNLOAD` has not been disabled and reinstall Argos Translate if necessary:

```bash
python -m pip install --upgrade argostranslate
```

### Port 8080 is already in use

```bash
export PORT=8081
python app.py
```

Then open `http://127.0.0.1:8081`.

## Security recommendations before remote exposure

Do not expose the built-in Flask server directly to a network. If remote access is required:

- Add authentication and authorization.
- Add CSRF protection and request-rate controls.
- Restrict upload types and sizes further.
- Isolate temporary files per authenticated user.
- Add session expiry and persistent session controls.
- Run behind a production WSGI server and a TLS-enabled reverse proxy.
- Review server-side request and prompt-injection risks.
- Store secrets in a proper secret manager.
- Add dependency scanning and version pinning.

## Licence

Published under the MIT license.

## Acknowledgements

This project integrates Flask, MLX, MLX-LM, MLX-VLM, MLX Whisper, Exa, the YouTube Transcript API, yt-dlp, Piper, NeMo Text Processing, Argos Translate, Pillow, and supporting Python libraries.
