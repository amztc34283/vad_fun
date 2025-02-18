# Turn Completion Detection Challenge (3 hours)

## Background
Voice Activity Detection (VAD) systems often cut off speakers mid-sentence because they rely on simple audio power thresholds (i.e. how high the signal is). Your task is to make a smarter system that can tell the difference between someone who's done talking versus just pausing to think.

## Task
Improve upon the basic VAD system to:
1. Detect true turn completions vs mid-sentence pauses
2. Work across multiple languages (English and Korean)
3. Process audio in near real-time

## Provided Materials
1. Starter code with basic VAD implementation
2. Test audio files:
   - `english_normal.wav`: Continuous English speech
   - `korean_normal.wav`: Continuous Korean speech  
   - `english_pause.wav`: English with "um" and mid-sentence pause
   - `korean_pause.wav`: Korean with mid-sentence pause

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements
Your solution should:
1. Implement `detect_turn_completion()` in `src/vad.py` to:
   - Handle mid-sentence pauses without cutting off
   - Detect filler words ("um", etc.)
   - Work for both English and Korean
   - Process audio chunks in near real-time

2. Consider:
   - What features beyond power help distinguish pauses vs completions?
   - How to track speaker state over time?
   - What makes a turn feel "complete" vs "paused"?
   - How to make this language-independent?

## Evaluation
Your solution will be evaluated on:
1. Technical Implementation:
   - Accuracy in detecting turns vs pauses
   - Language independence
   - Real-time processing capability

2. Code Quality:
   - Clear organization
   - Good documentation
   - Error handling

## Outside Resources
You are free to use any resources, including LLMs, papers, Stack Overflow as you wish. Please do this project alone without the help of another live human.

## Tips & Hints

* **Audio Features to Consider:**
  * Beyond simple power/energy, think about how pitch and rhythm change when someone is:
    * Finishing a thought
    * Pausing to think
    * Using filler words
  * What features might be universal across languages?

* **State Management:**
  * Consider keeping a short history of previous chunks
  * Think about different states a speaker might be in (speaking, paused, finished)
  * How long should a pause be before it's likely a turn completion?

* **Real-world Considerations:**
  * The system needs to make decisions quickly
  * You can't look too far ahead in the audio
  * False positives (cutting someone off) are worse than false negatives (waiting too long)

* **Testing Tips:**
  * Start with the normal files to get basic turn detection working
  * Use the pause files to test handling of hesitations
  * Compare behavior across languages to ensure language independence

## Submission
Provide:
1. Your completed code implementation
2. Brief write-up (1-2 paragraphs) explaining:
   - Your approach/strategy
   - Key features/decisions used
   - Challenges faced
   - Possible improvements with more time