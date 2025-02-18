# Turn Completion Detection Assignment (3 hours)

## Background
Current Voice Activity Detection (VAD) systems often use simple power thresholds to detect speech endpoints. However, this can lead to premature cutoffs during natural pauses or hesitations in speech.

## Task
Improve upon a basic VAD system to distinguish between:
- Natural turn completions (where the speaker is done)
- Mid-utterance pauses (where the speaker will continue)

## Provided Materials
1. Basic VAD implementation using power thresholds
2. Four test audio clips:
   - english_normal.wav: Continuous English speech
   - korean_normal.wav: Continuous Korean speech  
   - english_pause.wav: English with "um" and mid-sentence pause
   - korean_pause.wav: Korean with mid-sentence pause

## Requirements
1. Build upon the provided VAD code to handle:
   - Mid-sentence pauses
   - Filler words ("um", Korean equivalents)
   - Language-agnostic features
   
2. Your solution should:
   - Process audio in near real-time
   - Work for both English and Korean samples
   - Include comments explaining your approach

3. Evaluation metrics:
   - False cutoffs (cutting speaker off mid-sentence)
   - Latency (how long after true completion we detect it)
