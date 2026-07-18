from faster_whisper import WhisperModel
from typing import Optional

import re
import time


# Load the model only once when the backend starts.
model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)


BUSINESS_PROMPT = (
    "Business meeting, English, Vietnamese, software development, "
    "backend, frontend, API, database, deployment, deadline, "
    "project, client, budget, proposal, presentation, contract, "
    "strategy, revenue, marketing, stakeholder, meeting minutes."
)


# Common phrases Whisper may invent from silence or background noise.
HALLUCINATION_PHRASES = {
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subscribe to the channel",
    "you",
    "bye",
    "goodbye"
}


def normalize_text(text: str) -> str:
    """
    Clean spacing and repeated punctuation without changing
    meaningful English or Vietnamese words.
    """

    cleaned = re.sub(r"\s+", " ", text).strip()

    cleaned = re.sub(
        r"([.!?,])\1+",
        r"\1",
        cleaned
    )

    return cleaned


def looks_like_hallucination(text: str) -> bool:
    """
    Detect common short phrases produced from silence or noise.
    """

    normalized = text.lower().strip(" .,!?:;")

    if not normalized:
        return True

    if normalized in HALLUCINATION_PHRASES:
        return True

    # A one-character result is generally not useful speech.
    if len(normalized) <= 1:
        return True

    return False


def transcribe_audio(
    file_path: str,
    language_hint: Optional[str] = None
) -> dict:
    """
    Convert an audio file into text using Faster-Whisper.

    Returns the complete transcript, language information,
    timestamps, speech duration, confidence data, and
    performance measurements.
    """

    started_at = time.perf_counter()

    selected_language = None

    if language_hint in {"en", "vi"}:
        selected_language = language_hint

    segments_generator, info = model.transcribe(
        file_path,

        language=selected_language,

        beam_size=5,

        # Prevent unrelated chunks from influencing one another.
        condition_on_previous_text=False,

        # Reduce common silence hallucinations.
        hallucination_silence_threshold=1.0,

        # Ignore low-confidence tokens more aggressively.
        log_prob_threshold=-1.0,

        # Reject segments that resemble non-speech.
        no_speech_threshold=0.6,

        # Internal speech detection.
        vad_filter=True,

        vad_parameters={
            # Ignore extremely short sounds such as clicks.
            "min_speech_duration_ms": 250,

            # Allow natural short pauses within a sentence.
            "min_silence_duration_ms": 500,

            # Keep a small amount of surrounding audio.
            "speech_pad_ms": 200
        },

        initial_prompt=BUSINESS_PROMPT
    )

    transcript_parts = []
    timed_segments = []

    total_speech_duration = 0.0
    segment_probabilities = []

    for segment in segments_generator:
        text = normalize_text(
            segment.text
        )

        if not text:
            continue

        if looks_like_hallucination(text):
            continue

        segment_duration = max(
            0.0,
            float(segment.end - segment.start)
        )

        # Ignore tiny decoded fragments.
        if segment_duration < 0.15:
            continue

        total_speech_duration += segment_duration

        average_log_probability = float(
            getattr(segment, "avg_logprob", 0.0)
        )

        no_speech_probability = float(
            getattr(segment, "no_speech_prob", 0.0)
        )

        transcript_parts.append(text)

        segment_probabilities.append(
            average_log_probability
        )

        timed_segments.append({
            "start": round(
                float(segment.start),
                2
            ),

            "end": round(
                float(segment.end),
                2
            ),

            "duration": round(
                segment_duration,
                2
            ),

            "text": text,

            "average_log_probability": round(
                average_log_probability,
                4
            ),

            "no_speech_probability": round(
                no_speech_probability,
                4
            )
        })

    transcript = normalize_text(
        " ".join(transcript_parts)
    )

    processing_time_seconds = (
        time.perf_counter() - started_at
    )

    audio_duration_seconds = float(
        getattr(info, "duration", 0.0) or 0.0
    )

    if audio_duration_seconds > 0:
        real_time_factor = (
            processing_time_seconds
            / audio_duration_seconds
        )

        speech_ratio = (
            total_speech_duration
            / audio_duration_seconds
        )
    else:
        real_time_factor = None
        speech_ratio = 0.0

    if segment_probabilities:
        average_segment_log_probability = (
            sum(segment_probabilities)
            / len(segment_probabilities)
        )
    else:
        average_segment_log_probability = None

    # Require some actual speech instead of accepting
    # isolated noise or a very short hallucination.
    speech_detected = (
        bool(transcript)
        and total_speech_duration >= 0.25
    )

    if not speech_detected:
        transcript = ""
        timed_segments = []

    return {
        "transcript": transcript,

        "detected_language": info.language,

        "language_probability": float(
            info.language_probability
        ),

        "speech_detected": speech_detected,

        "segments": timed_segments,

        "audio_duration_seconds": round(
            audio_duration_seconds,
            2
        ),

        "speech_duration_seconds": round(
            total_speech_duration,
            2
        ),

        "speech_ratio": round(
            speech_ratio,
            3
        ),

        "processing_time_ms": round(
            processing_time_seconds * 1000,
            2
        ),

        "real_time_factor": (
            round(real_time_factor, 3)
            if real_time_factor is not None
            else None
        ),

        "average_segment_log_probability": (
            round(
                average_segment_log_probability,
                4
            )
            if average_segment_log_probability
            is not None
            else None
        )
    }